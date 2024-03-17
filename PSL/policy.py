import os
import cv2
import copy
import numpy as np
from typing import Dict, Optional, Tuple
from einops import rearrange

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import EigenCAM
from gym import spaces
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo.policy import Net, Policy
from habitat_baselines.utils.common import CategoricalNet, GaussianNet

from PSL.sensors import (
    ImageGoalSensorV2,
    CachedGoalSensor,
    ObjectGoalKShotImagePromptSensor,
    ObjectGoalPromptSensor,
    PanoramicImageGoalSensor,
    IINCachedGoalSensor,
    IINImageGoalSensor,
    ComplAttrCachedGoalSensor,
    QueriedImageGoalSensor,
    ObjectDistanceSensor,
    AugCachedGoalSensor,
)
from PSL.transforms import get_transform
from PSL.visual_encoder import VisualEncoder


def get_fg_clip_feature(model, image):
    def sub_forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)

        return x

    return sub_forward(model.visual, image.type(model.dtype))


def get_visual_encoder(backbone, baseplanes, pretrained_encoder, use_data_aug=True, run_type="train", enable_fg_midfusion=False, film_reduction="none"):
    name = "resize"
    if use_data_aug and run_type == "train":
        name = "resize+weak"
    spatial_size = 224 if enable_fg_midfusion else 128
    visual_transform = get_transform(name=name, size=spatial_size)
    cond_c = [256,] if enable_fg_midfusion else [1024,]
    visual_encoder = VisualEncoder(
        backbone=backbone, baseplanes=baseplanes, spatial_size=spatial_size,
        # film_reduction=film_reduction, film_layers=[0,], cond_c=cond_c, obs_c=[128,]
    )
    visual_size = visual_encoder.output_size

    if pretrained_encoder is not None:
        assert os.path.exists(pretrained_encoder)
        checkpoint = torch.load(pretrained_encoder, map_location="cpu")
        state_dict = checkpoint["teacher"]
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        msg = visual_encoder.load_state_dict(
            state_dict=state_dict, strict=False
        )
        logger.info("Using weights from {}: {}".format(pretrained_encoder, msg))

    return visual_encoder, visual_transform, visual_size


class ZSONPolicyNet(Net):
    def calculate_image_gradient(self, image):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        edge_x = F.conv2d(image, sobel_x.to(image.device), padding=1, groups=3)
        edge_y = F.conv2d(image, sobel_y.to(image.device), padding=1, groups=3)

        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        backbone: str,
        baseplanes: int,
        hidden_size: int,
        rnn_type: str,
        rnn_num_layers: int,
        use_data_aug: bool,
        use_clip_obs_encoder: bool,
        train_goal_encoder: bool,
        run_type: str,
        clip_model: str,
        obs_clip_model: str,
        pretrained_encoder: Optional[str] = None,
        enable_fg_midfusion: Optional[bool] = False,
        enable_sm_midfusion: Optional[bool] = False,
        film_reduction: Optional[str] = "none",
        use_layout_encoder: Optional[bool] = False,
        use_clip_corr_mapping: Optional[bool] = False,
        corr_hidden_size: Optional[int] = 512,
        gradient_image: Optional[bool] = False,
        cam_visual: Optional[bool] = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_clip_obs_encoder = use_clip_obs_encoder
        self.enable_fg_midfusion = enable_fg_midfusion
        self.enable_sm_midfusion = enable_sm_midfusion
        self.train_goal_encoder = train_goal_encoder
        self.use_layout_encoder = use_layout_encoder
        self.use_clip_corr_mapping = use_clip_corr_mapping
        self.corr_hidden_size = corr_hidden_size
        self.gradient_image = gradient_image
        self.cam_visual = cam_visual
        rnn_input_size = 0

        if self.gradient_image:
            for _ in range(10):
                logger.warning("Gradient image is used!")

        # ImageGoalSensorV2 is only for extracting lowlevel clip feature
        if ImageGoalSensorV2.cls_uuid in observation_space:
            assert self.enable_fg_midfusion

        # if sm_midfusion is enabled, one should perform global pooling
        if self.enable_sm_midfusion:
            assert film_reduction == "global"

        assert not (self.enable_fg_midfusion and self.enable_sm_midfusion)

        # visual encoder
        if self.use_clip_obs_encoder:
            name = "clip"
            if use_data_aug and run_type == "train":
                name = "clip+weak"
            self.visual_transform = get_transform(name=name, size=224)
            self.visual_encoder, _ = clip.load(obs_clip_model)
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
            self.visual_encoder.eval()
            visual_size = self.visual_encoder.visual.output_dim
            assert pretrained_encoder is None

            if self.use_clip_corr_mapping:
                self.policy_mlp = nn.Identity()
                self.corr_mapping_module = nn.Sequential(
                    nn.Linear(visual_size * 2, visual_size),
                    nn.ReLU(True),
                    nn.Linear(visual_size, self.corr_hidden_size),
                )
                rnn_input_size += self.corr_hidden_size
            else:
                # visual encoder mlp
                self.policy_mlp = nn.Sequential(
                    nn.Linear(visual_size, hidden_size // 2),
                    nn.ReLU(True),
                    nn.Linear(hidden_size // 2, hidden_size),
                )
                rnn_input_size += hidden_size

            self.visual_keys = [k for k in observation_space.spaces if k.startswith("rgb")]

        else:
            self.visual_encoder, self.visual_transform, visual_size = get_visual_encoder(
                backbone=backbone, baseplanes=baseplanes, 
                pretrained_encoder=pretrained_encoder, use_data_aug=use_data_aug,
                run_type=run_type, enable_fg_midfusion=enable_fg_midfusion, film_reduction=film_reduction
            )

            # visual encoder mlp
            self.policy_mlp = nn.Sequential(
                nn.Linear(visual_size, hidden_size // 2),
                nn.ReLU(True),
                nn.Linear(hidden_size // 2, hidden_size),
            )

            # update rnn input size
            self.visual_keys = [k for k in observation_space.spaces if k.startswith("rgb")]
            rnn_input_size += len(self.visual_keys) * hidden_size

        if self.use_layout_encoder:
            self.layout_encoder, self.layout_transform, layout_size = get_visual_encoder(
                backbone="resnet50", baseplanes=baseplanes, 
                pretrained_encoder=pretrained_encoder, use_data_aug=use_data_aug,
                run_type=run_type,
            )
            # visual encoder mlp
            # self.layout_policy_mlp = nn.Sequential(
            #     nn.Linear(layout_size, hidden_size // 2),
            #     nn.ReLU(True),
            #     nn.Linear(hidden_size // 2, hidden_size),
            # )
            self.layout_policy_mlp = nn.Linear(layout_size, 128)
            rnn_input_size += 128

        # goal embedding
        goal_uuids = [
            PanoramicImageGoalSensor.cls_uuid,
            ObjectGoalPromptSensor.cls_uuid,
            CachedGoalSensor.cls_uuid,
            IINCachedGoalSensor.cls_uuid,
            ObjectGoalKShotImagePromptSensor.cls_uuid,
            ComplAttrCachedGoalSensor.cls_uuid,
            QueriedImageGoalSensor.cls_uuid,
            ImageGoalSensorV2.cls_uuid,
            IINImageGoalSensor.cls_uuid,
            AugCachedGoalSensor.cls_uuid,
        ]
        goal_uuid = [uuid for uuid in observation_space.spaces if uuid in goal_uuids]
        # assert len(goal_uuid) == 1
        if len(goal_uuid) > 1:
            logger.warning("multiple goal sensor: {}".format(goal_uuid))
        goal_uuid = goal_uuid[0]
        self.goal_uuid = goal_uuid

        # CLIP goal encoder
        if self.train_goal_encoder:
            assert self.goal_uuid in [ImageGoalSensorV2.cls_uuid, IINImageGoalSensor.cls_uuid, QueriedImageGoalSensor.cls_uuid]
            self.goal_encoder, self.goal_transform, _ = get_visual_encoder(
                backbone=backbone, baseplanes=baseplanes, 
                pretrained_encoder=pretrained_encoder, use_data_aug=use_data_aug,
                run_type=run_type, enable_fg_midfusion=enable_fg_midfusion, film_reduction=film_reduction
            )
            self.goal_policy_mlp = nn.Sequential(
                nn.Linear(visual_size, hidden_size // 2),
                nn.ReLU(True),
                nn.Linear(hidden_size // 2, hidden_size),
            )
            goal_size = hidden_size
        elif goal_uuid != CachedGoalSensor.cls_uuid or self.enable_fg_midfusion:
            # assert run_type == "eval"
            self.clip_transform = get_transform(name="clip", size=224)
            if clip_model.lower() == "siglip":
                from open_clip import create_model_from_pretrained
                self.clip, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
                goal_size = 768
            else:
                self.clip, _ = clip.load(clip_model)
                goal_size = self.clip.visual.output_dim
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()
        else:
            goal_size = 1024 if clip_model == "RN50" else 768

        if self.use_clip_corr_mapping:
            goal_size = 0

        # goal embedding size
        rnn_input_size += goal_size

        # previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        rnn_input_size += 32

        # other sensor input
        if ObjectDistanceSensor.cls_uuid in observation_space.spaces:
            n_distance_level = 11 # [TODO] write to config
            self.distance_embedding = nn.Embedding(n_distance_level, 32)
            rnn_input_size += 32

        # state encoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=rnn_num_layers,
        )

    @property
    def output_size(self):
        return self.hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
    
    def visualize_cam(self, inputs, model, layer, images, visual=False):
        visualization = []
        activation = []
        cam = EigenCAM(model=model, target_layers=[layer], use_cuda=True)
        output = cam(inputs)
        for i, v in enumerate(output):
            h, w = images[i].shape[:2]
            v = cv2.resize(v, (w, h))
            activation.append(copy.deepcopy(v))

            # visualization
            if visual:
                v = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
                v = cv2.addWeighted(v, 0.5, images[i].cpu().numpy(), 0.5, 0)
                visualization.append(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        
        visualization = np.stack(visualization, axis=0) if visual else None
        activation = np.stack(activation, axis=0)

        return visualization, activation

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """In this function `N` will denote the number of environments, `T` the
        number of timesteps, `V` the number of views and `K` is number of shots.
        """

        x = []

        # stack rgb observations
        obs = torch.stack([observations[k] for k in self.visual_keys], dim=1)

        # get shapes
        TN = obs.size(0)
        N = rnn_hidden_states.size(0)
        T = TN // N
        OV = obs.size(1)

        goal = None
        if self.enable_fg_midfusion:
            fg_goal = observations[ImageGoalSensorV2.cls_uuid] # [FIXME]
            with torch.no_grad():
                fg_goal = self.clip_transform(goal, T, N, 1)
                midfusion_feature = get_fg_clip_feature(self.clip, fg_goal).float()
        elif self.enable_sm_midfusion:
            if self.goal_uuid == ObjectGoalPromptSensor.cls_uuid:
                goal = observations[self.goal_uuid]  # TN x 1 x F
                GV = goal.size(1)
                goal = goal.flatten(0, 1)  # TN * GV x F
                with torch.no_grad():
                    if len(goal.shape) == 3:
                        B, L, N = goal.shape
                        goal = rearrange(goal, "b l n -> (b n) l")
                        goal = self.clip.encode_text(goal).float()
                        goal = rearrange(goal, "(b n) l -> b l n", b=B, n=N).mean(dim=2)
                    else:
                        goal = self.clip.encode_text(goal).float()
                    goal /= goal.norm(dim=-1, keepdim=True)
            else:
                goal = observations[self.goal_uuid].float()
                GV = goal.size(1)
                goal = goal.flatten(0, 1)  # TN * GV x D
                goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D

            midfusion_feature = goal.unsqueeze(-1).unsqueeze(-1)

        # visual encoder
        obs = obs.flatten(0, 1)  # TN * OV x H x W x 3
        rgb = self.visual_transform(obs, T, N, OV)  # TN * OV x h x w x 3
        if self.gradient_image:
            rgb = self.calculate_image_gradient(rgb)
        if self.use_clip_obs_encoder:
            rgb = self.visual_encoder.encode_image(rgb).float()
        elif self.enable_fg_midfusion or self.enable_sm_midfusion:
            rgb = self.visual_encoder(rgb, [midfusion_feature,])
        else:
            rgb = self.visual_encoder(rgb)  # TN * OV x D
        rgb = self.policy_mlp(rgb)  # TN * OV x d
        rgb = rgb.reshape(TN, OV, -1)  # TN x OV x d
        rgb = rgb.flatten(1)  # TN x OV * d
        x.append(rgb)

        if self.use_layout_encoder:
            layout = self.layout_transform(obs, T, N, OV)
            z_layout = self.layout_encoder(layout)
            z_layout = self.layout_policy_mlp(z_layout)  # TN * OV x d
            z_layout = z_layout.reshape(TN, OV, -1)  # TN x OV x d
            z_layout = z_layout.flatten(1)  # TN x OV * d
            x.append(z_layout)

        # goal embedding
        if CachedGoalSensor.cls_uuid in observations:
            goal = observations[CachedGoalSensor.cls_uuid].float()
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D

        if AugCachedGoalSensor.cls_uuid in observations:
            goal = observations[AugCachedGoalSensor.cls_uuid].float()
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D

        if PanoramicImageGoalSensor.cls_uuid in observations:
            goal = observations[PanoramicImageGoalSensor.cls_uuid]
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x H x W x 3
            with torch.no_grad():
                goal = self.clip_transform(goal, T, N, GV)
                goal = self.clip.encode_image(goal).float()
                goal /= goal.norm(dim=-1, keepdim=True)

        if IINCachedGoalSensor.cls_uuid in observations:
            goal = observations[IINCachedGoalSensor.cls_uuid].float()
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D

        if QueriedImageGoalSensor.cls_uuid in observations:
            goal = observations[QueriedImageGoalSensor.cls_uuid].float()
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D

        if ObjectGoalPromptSensor.cls_uuid in observations:
            goal = observations[ObjectGoalPromptSensor.cls_uuid]  # TN x 1 x F
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x F
            with torch.no_grad():
                if len(goal.shape) == 3:
                    B, L, N = goal.shape
                    goal = rearrange(goal, "b l n -> (b n) l")
                    goal = self.clip.encode_text(goal).float()
                    goal = rearrange(goal, "(b n) l -> b l n", b=B, n=N).mean(dim=2)
                else:
                    goal = self.clip.encode_text(goal).float()
                goal /= goal.norm(dim=-1, keepdim=True)

        if ObjectGoalKShotImagePromptSensor.cls_uuid in observations:
            goal = observations[ObjectGoalKShotImagePromptSensor.cls_uuid]  # TN x K x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN x K x D
            with torch.no_grad():
                rgb = self.clip_transform(obs, T, N, OV)  # TN * OV x 3 x h x w
                rgb = self.clip.encode_image(rgb).float()  # TN * OV x D
                rgb /= rgb.norm(dim=-1, keepdim=True)  # TN * OV x D
            # assume V=1 for our case
            cosine_similarity = torch.einsum("nkd,nd->nk", goal, rgb).unsqueeze(
                dim=-1
            )  # TN x K x 1
            goal_weights = F.softmax(cosine_similarity, dim=1)  # TN x K x 1
            goal = torch.sum(goal * goal_weights, dim=1, keepdim=True)  # TN x 1 x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN x 1 x D
            GV = 1

        if ComplAttrCachedGoalSensor.cls_uuid in observations:
            compl_attr = observations[ComplAttrCachedGoalSensor.cls_uuid].float()
            GV = compl_attr.size(1)
            compl_attr = compl_attr.flatten(0, 1)  # TN * GV x D
            compl_attr /= compl_attr.norm(dim=-1, keepdim=True)  # TN * GV x D
            if goal is None:
                goal = compl_attr
            else:
                goal = (goal + compl_attr) / 2 # add on

        if self.train_goal_encoder:
            if QueriedImageGoalSensor.cls_uuid in observations:
                goal = observations[QueriedImageGoalSensor.cls_uuid].float()
                GV = goal.size(1)
                goal = goal.flatten(0, 1)  # TN * GV x D
                # do not need to norm
                # goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D
            else:
                assert ImageGoalSensorV2.cls_uuid in observations or \
                    IINImageGoalSensor.cls_uuid in observations
                assert not (ImageGoalSensorV2.cls_uuid in observations and \
                    IINImageGoalSensor.cls_uuid in observations)
                image_goal_sensor_uuid = ImageGoalSensorV2.cls_uuid if ImageGoalSensorV2.cls_uuid in observations else IINImageGoalSensor.cls_uuid
                goal = observations[image_goal_sensor_uuid].to(torch.uint8) # TN * H * W * 3
                goal = self.goal_transform(goal, T, N, OV)  # TN * OV x h x w x 3
                if self.gradient_image:
                    goal = self.calculate_image_gradient(goal)

                goal = self.goal_encoder(goal)  # TN * OV x D
                goal = self.goal_policy_mlp(goal)  # TN * OV x d
                goal = goal.unsqueeze(1) # TN x GV x D
                GV = 1

        # average pool goal embedding (Note: this is a no-op for object goal
        # representations and single view goals because GV == 1)
        goal = goal.reshape(TN, GV, -1)  # TN x GV x D
        goal = goal.mean(1)  # TN x D

        x.append(goal)

        if self.use_layout_encoder:
            assert len(x) == 3 # rgb, layout, goal
            x = [self.corr_mapping_module(torch.concat(
                [x[0], x[-1]], dim=-1
            )), x[1]]

        # previous action embedding
        prev_actions = prev_actions.squeeze(-1)  # TN
        start_token = torch.zeros_like(prev_actions)  # TN
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )  # TN x 32
        x.append(prev_actions)  # TN x 32

        # other sensor
        if ObjectDistanceSensor.cls_uuid in observations:
            object_distance = observations[ObjectDistanceSensor.cls_uuid]
            object_distance = object_distance.squeeze(-1)  # TN
            object_distance = self.distance_embedding(object_distance)  # TN x 32
            x.append(object_distance)  # TN x 32

        # state encoding
        rnn_input = torch.cat(x, dim=1)
        prev_rnn_hidden_states = rnn_hidden_states
        out, rnn_hidden_states = self.state_encoder(rnn_input, prev_rnn_hidden_states, masks)

        # gradcam visualization
        if self.cam_visual:
            cam_visual, cam_activation = self.visualize_cam(
                layout, 
                self.layout_encoder, 
                self.layout_encoder.backbone.__getattr__("layer3"), 
                obs,
                True,
            )
            observations['cam_visual'] = torch.from_numpy(cam_visual).cuda()

        return out, rnn_hidden_states


@baseline_registry.register_policy
class ZSONPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        backbone: str = "resnet18",
        baseplanes: int = 32,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        rnn_num_layers: int = 1,
        use_data_aug: bool = False,
        run_type: str = "train",
        clip_model: str = "RN50",
        obs_clip_model: str = "RN50",
        pretrained_encoder: Optional[str] = None,
        use_clip_obs_encoder: bool = False,
        train_goal_encoder: bool = False,
        enable_fg_midfusion: bool = False,
        enable_sm_midfusion: bool = False,
        film_reduction: str = "none",
        use_layout_encoder: bool = False,
        use_clip_corr_mapping: bool = False,
        corr_hidden_size: int = 512,
        gradient_image: bool = False,
        cam_visual: bool = False,
    ):
        super().__init__(
            net=ZSONPolicyNet(
                observation_space=observation_space,
                action_space=action_space,
                backbone=backbone,
                baseplanes=baseplanes,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                rnn_num_layers=rnn_num_layers,
                use_data_aug=use_data_aug,
                run_type=run_type,
                clip_model=clip_model,
                obs_clip_model=obs_clip_model,
                pretrained_encoder=pretrained_encoder,
                use_clip_obs_encoder=use_clip_obs_encoder,
                train_goal_encoder=train_goal_encoder,
                enable_fg_midfusion=enable_fg_midfusion,
                enable_sm_midfusion=enable_sm_midfusion,
                film_reduction=film_reduction,
                use_layout_encoder=use_layout_encoder,
                use_clip_corr_mapping=use_clip_corr_mapping,
                corr_hidden_size=corr_hidden_size,
                gradient_image=gradient_image,
                cam_visual=cam_visual,
            ),
            dim_actions=action_space.n,
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            backbone=config.RL.POLICY.backbone,
            baseplanes=config.RL.POLICY.baseplanes,
            hidden_size=config.RL.POLICY.hidden_size,
            rnn_type=config.RL.POLICY.rnn_type,
            rnn_num_layers=config.RL.POLICY.rnn_num_layers,
            use_data_aug=config.RL.POLICY.use_data_aug,
            run_type=config.RUN_TYPE,
            clip_model=config.RL.POLICY.CLIP_MODEL,
            obs_clip_model=config.RL.POLICY.OBS_CLIP_MODEL,
            pretrained_encoder=config.RL.POLICY.pretrained_encoder,
            use_clip_obs_encoder=config.RL.POLICY.use_clip_obs_encoder,
            train_goal_encoder=config.RL.POLICY.train_goal_encoder,
            enable_fg_midfusion=config.RL.POLICY.enable_fg_midfusion,
            enable_sm_midfusion=config.RL.POLICY.enable_sm_midfusion,
            film_reduction=config.RL.POLICY.film_reduction,
            use_layout_encoder=config.RL.POLICY.use_layout_encoder,
            use_clip_corr_mapping=config.RL.POLICY.use_clip_corr_mapping,
            corr_hidden_size=config.RL.POLICY.corr_hidden_size,
            gradient_image=config.RL.POLICY.gradient_image,
            cam_visual=config.RL.POLICY.cam_visual,
        )
