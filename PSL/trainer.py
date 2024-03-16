import copy
import json
import os
from typing import Any, Dict, List, Tuple
from collections import defaultdict, deque
import random
import time

import ifcfg
import attr
import numpy as np
import torch
import tqdm
from gym import spaces
from PIL import Image
from habitat import Config, logger, make_dataset
from habitat.utils import profiling_wrapper
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    # observations_to_image,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    load_resume_state,
    rank0_only,
    get_distrib_size,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
    generate_video,
)
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)
from torch import nn

from PSL.ppo import ZSON_DDPPO, ZSON_PPO
from PSL.visualization import observations_to_image


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def get_episode_json(episode, reference_replay):
    ep_json = attr.asdict(episode)
    ep_json["trajectory"] = reference_replay
    return ep_json


def init_distrib_nccl(
    backend: str = "nccl",
) -> Tuple[int, torch.distributed.TCPStore]:  # type: ignore
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when ``srun`` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = ifcfg.default_interface()["device"]

    local_rank, world_rank, world_size = get_distrib_size()

    main_port = int(os.environ.get("MASTER_PORT", 16384))
    main_addr = str(os.environ.get("MASTER_ADDR", "127.0.0.1"))

    if world_rank == 0:
        logger.info('distributed url: {}:{}'.format(main_addr, main_port))

    tcp_store = torch.distributed.TCPStore(  # type: ignore
        main_addr, main_port + 1, world_size, world_rank == 0
    )
    torch.distributed.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store


@baseline_registry.register_trainer(name="zson-ddppo")
@baseline_registry.register_trainer(name="zson-ppo")
class ZSONTrainer(PPOTrainer):
    def _init_train(self):
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if self._is_distributed:
            local_rank, world_rank, world_size = get_distrib_size()

            local_rank, tcp_store = init_distrib_nccl(
                self.config.RL.DDPPO.distrib_backend
            )

            logger.info(
                "initialize ddp done: local rank %02d | world rank %02d | world size %02d" 
                    % (local_rank, world_rank, world_size)
            )
            
            torch.distributed.barrier()
            
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )
                
            resume_state = load_resume_state(self.config)
            if resume_state is not None:
                self.config: Config = resume_state["config"]
                self.using_velocity_ctrl = (
                    self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
                ) == ["VELOCITY_CONTROL"]
                del resume_state

            self.config.defrost()

            if "*" in self.config.TASK_CONFIG.DATASET.CONTENT_SCENES:
                dataset = make_dataset(self.config.TASK_CONFIG.DATASET.TYPE)
                scenes = dataset.get_scenes_to_load(self.config.TASK_CONFIG.DATASET)
                random.shuffle(scenes)
                
                if len(scenes) >= 200:
                    scene_splits: List[List[str]] = [[] for _ in range(world_size)]
                    for idx, scene in enumerate(scenes):
                        scene_splits[idx % len(scene_splits)].append(scene)

                    assert sum(map(len, scene_splits)) == len(scenes)

                    for i in range(world_size):
                        if len(scenes) > 0:
                            self.config.TASK_CONFIG.DATASET.CONTENT_SCENES = scene_splits[i]

            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        else:
            raise NotImplementedError("Do not try to train the model without distributed mode")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0][
                "VELOCITY_CONTROL"
            ]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = None
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self.agent = self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=False)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
        
        torch.distributed.barrier()

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        if rank0_only():
            logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        def preprocess_pretrained_weight(state_dict):
            state_dict = {
                k[len("actor_critic.") :]: v
                for k, v in state_dict.items()
            }
            if state_dict["net.prev_action_embedding.weight"].shape != \
                self.actor_critic.net.prev_action_embedding.weight.shape:
                state_dict = {
                    k : v
                    for k, v in state_dict.items()
                    if k not in ["net.prev_action_embedding.weight", 
                                 "action_distribution.linear.weight",
                                 "action_distribution.linear.bias"]
                }
            return state_dict

        if self.config.RL.DDPPO.pretrained:
            msg = self.actor_critic.load_state_dict(
                preprocess_pretrained_weight(pretrained_state["state_dict"]),
                strict=False
            )
            print(msg)
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        return (ZSON_DDPPO if self._is_distributed else ZSON_PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            wd=ppo_cfg.wd,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    METRICS_BLACKLIST = {
        "top_down_map",
        "collisions.is_collision",
        "agent_position",
        "agent_rotation",
    }

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(
                    self.config.EVAL_CKPT_PATH_DIR
                )
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = self.config.EVAL_CKPT_START_ID
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")  # type: ignore
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            logger.info("loading from")
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0]["VELOCITY_CONTROL"]
            action_shape = (2,)
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long

        self.agent = self._setup_actor_critic_agent(ppo_cfg)

        msg = self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
        logger.info(msg)
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        text_queries = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        evaluation_meta = []
        ep_actions = [[] for _ in range(self.config.NUM_ENVIRONMENTS)]
        possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        # create a partial aggregated logger
        partial_aggregated_stats = {}

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (_, actions, _, test_recurrent_hidden_states,) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=config.DETERMINISTIC,
                )

                prev_actions.copy_(actions)  # type: ignore
            action_names = [
                possible_actions[a.item()] for a in actions.to(device="cpu")
            ]
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if self.using_velocity_ctrl:
                step_data = [
                    action_to_velocity_control(a) for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            if "cam_visual" in batch:
                cam_visual = batch["cam_visual"]
            else:
                cam_visual = None

            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            if cam_visual is not None:
                batch["cam_visual"] = cam_visual

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0

                    for stat_key in episode_stats.keys():
                        if stat_key not in partial_aggregated_stats:
                            partial_aggregated_stats[stat_key] = episode_stats[stat_key]
                        else:
                            partial_aggregated_stats[stat_key] = (
                                (partial_aggregated_stats[stat_key] * (pbar.n - 1) + episode_stats[stat_key]) / pbar.n
                            )

                    if pbar.n % 100 == 0 and pbar.n > 0:
                        print("\n".join([f"{k}: {v}" for k,v in partial_aggregated_stats.items()]))

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            # episode_id=current_episodes[i].episode_id,
                            episode_id="{}_{}".format(
                                current_episodes[i].scene_id.rsplit("/", 1)[-1],
                                current_episodes[i].episode_id,
                            ),
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )
                        rgb_frames[i] = []
                    elif self.config.SAVE_LAST_OBS and len(rgb_frames[i]):
                        # print(rgb_frames[-1].shape)
                        episode_id = "{}_{}".format(
                            current_episodes[i].scene_id.rsplit("/", 1)[-1],
                            current_episodes[i].episode_id,
                        )
                        dump_root = os.path.join(os.path.dirname(self.config.CHECKPOINT_FOLDER), "results")
                        os.makedirs(dump_root, exist_ok=True)
                        Image.fromarray(rgb_frames[i][-1]).save(f"{dump_root}/{episode_id}.png")
                        with open(f"{dump_root}/{episode_id}.json", "w") as fo:
                            json.dump({
                                "text": text_queries[i][-1],
                                "goal":current_episodes[i].goals_key
                            }, fo)
                        # print(rgb_frames[i][-1].shape)
                        # print(text_queries[i][-1])

                # episode continues
                else:
                    if len(self.config.VIDEO_OPTION) > 0:
                        # TODO move normalization / channel changing out of the policy and undo it here
                        frame = observations_to_image(
                            {k: v[i] for k, v in batch.items()}, infos[i]
                        )
                        frame = append_text_to_image(
                            frame,
                            "Intrinsic: {}, Extrinsic: {}".format(
                                current_episodes[i].goal_attributes["intrinsic_attributes"],
                                current_episodes[i].goal_attributes["extrinsic_attributes"]
                            ) if getattr(current_episodes[i], "goal_attributes", None) is not None else \
                            "Find and go to {}".format(
                                current_episodes[i].object_category
                            ),
                        )
                        rgb_frames[i].append(frame)
                    elif self.config.SAVE_LAST_OBS:
                        rgb_frames[i].append(batch["rgb"][i].cpu().numpy())
                        text_queries[i].append(
                            ". ".join([
                                current_episodes[i].goal_attributes["intrinsic_attributes"].split(".")[0],
                                current_episodes[i].goal_attributes["extrinsic_attributes"].split(".")[0],
                            ])
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
