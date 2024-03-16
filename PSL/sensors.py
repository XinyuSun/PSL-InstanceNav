from typing import Any, Optional
import os
import clip
import nltk
import torch
import math
import numpy as np
import cv2
from gym import Space, spaces
from PIL import Image
import quaternion
import magnum as mn

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.utils.geometry_utils import quaternion_from_coeff

from habitat import logger

from PSL.dataset import AttrObjNavEpisode


@registry.register_sensor
class ImageGoalSensorV2(Sensor):
    cls_uuid: str = "imagegoal_sensor_v2"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalSensorV2 requires one RGB sensor, "
                f"{len(rgb_sensor_uuids)} detected"
            )

        # (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_scene_id = None
        self._current_episode_id = None
        self._current_image_goal = None
        self._sampling_type = getattr(config, 'SAMPLE_TYPE', 'uniform')
        channels = getattr(config, 'CHANNELS', ['rgb'])
        if isinstance(channels, list):
            self._channels = channels
        elif isinstance(channels, str):
            # string with / to separate modalities
            self._channels = channels.split('/')
        else:
            raise ValueError(f'Unknown data type for channels!')

        self._channel2uuid = {}
        self._channel2range = {}
        self._shape = None
        self._current_goal_views = []
        self._setup_channels()
        self._set_space()
        super().__init__(config=config)

    def _get_sensor_uuid(self, sensor_type):
        sensors = self._sim.sensor_suite.sensors
        sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, sensor_type)
        ]
        if len(sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalSensorV2 requires one {sensor_type} sensor, "
                f"{len(sensor_uuids)} detected"
            )

        return sensor_uuids[0]

    def _setup_channels(self):
        self._channel2uuid = {}
        self._channel2range = {}
        last_idx = 0
        if 'rgb' in self._channels:
            self._channel2uuid['rgb'] = self._get_sensor_uuid(RGBSensor)
            self._channel2range['rgb'] = (last_idx, last_idx + 3)
            last_idx += 3

        if len(self._channel2uuid.keys()) == 0:
            raise ValueError('ImageGoalSensorV2 requires at least one channel')

    def _set_space(self):
        self._shape = None
        for k in self._channel2uuid.keys():
            uuid = self._channel2uuid[k]
            ospace = self._sim.sensor_suite.observation_spaces.spaces[uuid]
            if self._shape is None:
                self._shape = [ospace.shape[0], ospace.shape[1], 0]
            else:
                if ((self._shape[0] != ospace.shape[0]) or
                    (self._shape[1] != ospace.shape[1])):
                    raise ValueError('ImageGoalSensorV2 requires all '
                                     'base sensors to have the same with '
                                     'and hight, {uuid} has shape {ospace.shape}')

            if len(ospace.shape) == 3:
                self._shape[2] += ospace.shape[2]
            else:
                self._shape[2] += 1

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=self._shape,
            dtype=np.float32)

    def _get_image_goal_at(self, position, rotation):
        obs = self._sim.get_observations_at(
            position=position, rotation=rotation)
        goal = []
        if 'rgb' in self._channel2uuid.keys():
            goal.append(obs[self._channel2uuid['rgb']].astype(
                self.observation_space.dtype))

        return np.concatenate(goal, axis=2)

    def _get_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        self._current_goal_views = []
        
        # default episode angle
        seed = abs(hash(episode.episode_id)) % (2**32)  # deterministic angle
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)

        obs_idx = 1  # default observation index
        if self.config.SINGLE_VIEW and self.config.SAMPLE_GOAL_ANGLE:
            # sample an observation index and offset the default angle
            obs_idx = np.random.choice(4)
            angle += [1 / 2 * np.pi, 0.0, 3 / 2 * np.pi, np.pi][obs_idx]

        # set the goal rotation
        goal_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = quaternion_from_coeff(goal_rotation)
        
        goal_observation = self._get_image_goal_at(
            goal_position.tolist(), goal_rotation)
        self._current_goal_views.append(goal_rotation)

        return goal_observation

    def get_observation(
        self, *args: Any, observations, episode: NavigationEpisode, **kwargs: Any
    ):
        if episode.scene_id != self._current_scene_id:
            self._current_scene_id = episode.scene_id

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class PanoramicImageGoalSensor(Sensor):
    cls_uuid: str = "panoramic-imagegoal"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        self._uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        h, w, d = self._sim.sensor_suite.observation_spaces.spaces[self._uuids[0]].shape
        return spaces.Box(
            low=0, high=255, shape=(len(self._uuids), h, w, d), dtype=np.uint8
        )

    def _get_panoramic_image_goal(self, episode: NavigationEpisode):
        goal_position = list(episode.goals[0].position)

        if self.config.SAMPLE_GOAL_ANGLE:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            seed = abs(hash(episode.episode_id)) % (2**32)  # deterministic angle
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)

        # set the goal rotation
        goal_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = quaternion_from_coeff(goal_rotation)

        # get the goal observation
        goal_observation = self._sim.get_observations_at(
            position=goal_position, rotation=goal_rotation
        )
        return np.stack([goal_observation[k] for k in self._uuids])

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_panoramic_image_goal(episode)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class ObjectGoalPromptSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoalprompt"

    def __init__(
        self,
        *args: Any,
        config: Config,
        **kwargs: Any,
    ):
        super().__init__(config=config)
        if config.CLIP_MODEL.lower() == "siglip":
            from open_clip import get_tokenizer
            self.tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
            self.context_length = 64
        else:
            self.tokenizer = clip.tokenize
            self.context_length = 77

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=np.inf, shape=(77,), dtype=np.int64)

    def get_observation(
        self,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        category = episode.object_category if hasattr(episode, "object_category") else ""
        if self.config.HAS_ATTRIBUTE:
            intrinsic_attributes = episode.goal_attributes["intrinsic_attributes"].split(".")[0]
            if self.config.PARSE:
                tokens = nltk.word_tokenize(intrinsic_attributes)
                pos_tags = nltk.pos_tag(tokens)
                nounes = [pt[0] for pt in pos_tags if pt[1].startswith("N")]
                attr = [pt[0] for pt in pos_tags if pt[1].startswith("J")]
                intrinsic_attributes = " ".join(attr + nounes)
            intrinsic_attributes = intrinsic_attributes.strip()
            extrinsic_attributes = episode.goal_attributes["extrinsic_attributes"].split(".")[0]
            feat_int = self.tokenizer(intrinsic_attributes, context_length=self.context_length).numpy()
            feat_ext = self.tokenizer(extrinsic_attributes, context_length=self.context_length).numpy()

            if self.config.ATTRIBUTE_TYPE == "both":
                return np.stack((feat_int, feat_ext), axis=2)
            elif self.config.ATTRIBUTE_TYPE == "intrinsic_attributes":
                return feat_int
            elif self.config.ATTRIBUTE_TYPE == "extrinsic_attributes":
                return feat_ext
            else:
                raise TypeError
            # prompt = intrinsic_attributes + extrinsic_attributes
            # tokens = category.split("_")
            # attr, cat = tokens[0], " ".join(tokens[1:])  # assume one word attributes
            
            # return np.mean(np.stack((feat_int, feat_ext)), axis=0)
            
        else:
            attr, cat = None, category.replace("_", " ")
            # use `attr` and `cat` in prompt templates
            prompt = self.config.PROMPT.format(attr=attr, cat=cat)
            return self.tokenizer(prompt, context_length=self.context_length).numpy()


@registry.register_sensor
class CachedGoalSensor(Sensor):
    cls_uuid: str = "cached-goal"

    def __init__(self, *args: Any, config: Config, dataset: Dataset, **kwargs: Any):
        self._current_episode_id: Optional[str] = None
        self._current_goal = None

        self._data = {}
        self._angle = {}
        for scene_id in dataset.scene_ids:
            scene = dataset.scene_from_scene_path(scene_id)
            path = config.DATA_PATH.format(split=config.DATA_SPLIT, scene=scene)
            self._data[scene_id] = np.load(path, mmap_mode="r")
            if os.path.exists(path.replace(".npy", ".angle.npy")):
                self._angle[scene_id] = np.load(path.replace(".npy", ".angle.npy"), mmap_mode="r")

        super().__init__(config=config)

    def _get_goal(self, episode):
        """order: left, front, right, back."""
        scene_id, episode_id = episode.scene_id, episode.episode_id
        obs = self._data[scene_id][episode_id]

        # default episode angle
        if len(self._angle) and scene_id in self._angle:
            obs_idx = 0  # default observation index
            angle = float(self._angle[scene_id][episode_id][obs_idx])
        else:
            seed = abs(hash(episode.episode_id)) % (2**32)  # deterministic angle
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
            obs_idx = 1  # default observation index

        if self.config.SINGLE_VIEW and self.config.SAMPLE_GOAL_ANGLE:
            # sample an observation index and offset the default angle
            obs_idx = np.random.choice(4)
            angle += [1 / 2 * np.pi, 0.0, 3 / 2 * np.pi, np.pi][obs_idx]

        # set the goal rotation
        goal_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = quaternion_from_coeff(goal_rotation)

        # return the observation embedding
        if self.config.SINGLE_VIEW:
            return obs[obs_idx][None]
        else:
            return obs

    def _get_example_observation(self):
        scene_id = list(self._data.keys())[0]
        obs = self._data[scene_id][0]
        if self.config.SINGLE_VIEW:
            return obs[0][None]
        else:
            return obs

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        obs = self._get_example_observation()
        return spaces.Box(
            low=float("-inf"), high=float("inf"), shape=obs.shape, dtype=obs.dtype
        )

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_goal

        self._current_episode_id = episode_uniq_id
        self._current_goal = self._get_goal(episode)

        return self._current_goal


@registry.register_sensor
class ObjectGoalKShotImagePromptSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will use sample object image prompt corresponding to it
    so that it's usable by CLIP's image encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalKShotImagePromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal_kshot_image_prompt"

    def __init__(
        self,
        *args: Any,
        config: Config,
        **kwargs: Any,
    ):

        self._data = {}
        self._data = np.load(config.DATA_PATH, allow_pickle=True).item()
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.SHOTS, self.config.CLIP_SIZE),
            dtype=np.float32,
        )

    def get_observation(
        self,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        if not hasattr(episode, "object_category"):
            return np.zeros((self.config.SHOTS, self.config.CLIP_SIZE))
        image_goal_samples = self._data[episode.object_category]
        if self.config.SHOTS >= len(image_goal_samples):
            return image_goal_samples
        shots_ind = np.random.permutation(len(image_goal_samples))[: self.config.SHOTS]
        return image_goal_samples[shots_ind]


@registry.register_sensor
class ComplAttrCachedGoalSensor(Sensor):
    r"""
    """
    cls_uuid: str = "compl_attr_cached_goal"

    def __init__(self, *args: Any, config: Config, dataset: Dataset, **kwargs: Any):
        self._current_goal_id: Optional[str] = None

        self._data = torch.load(config.DATA_PATH.format(split=config.DATA_SPLIT))
        
        super().__init__(config=config)

    def _get_example_observation(self):
        object_id = list(self._data.keys())[0]
        obs = self._data[object_id]
        return obs

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.CLIP_SIZE,),
            dtype=np.float32,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_goal(self, id):
        return self._data[id].numpy()

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        scene_short_id = os.path.basename(episode.scene_id).split(".")[0]
        goal_object_id = episode.goal_object_id
        goal_id = f"{scene_short_id}_{goal_object_id}"
        if goal_id == self._current_goal_id:
            return self._current_goal

        self._current_goal_id = goal_id
        self._current_goal = self._get_goal(goal_id)

        return self._current_goal


@registry.register_sensor
class QueriedImageGoalSensor(ComplAttrCachedGoalSensor):
    cls_uuid: str = "queried-image-goal"

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        if isinstance(episode, AttrObjNavEpisode):
            scene_short_id = os.path.basename(episode.scene_id).split(".")[0]
            goal_object_id = episode.goal_object_id
            goal_id = f"{scene_short_id}_{goal_object_id}"
        elif isinstance(episode, ObjectGoalNavEpisode):
            goal_id = episode.object_category
        else:
            raise TypeError("Unsupported episode type!")
        
        if goal_id == self._current_goal_id:
            return self._current_goal

        self._current_goal_id = goal_id
        self._current_goal = self._get_goal(goal_id)

        return self._current_goal


@registry.register_sensor
class IINCachedGoalSensor(Sensor):
    cls_uuid: str = "iin-cached-goal"

    def __init__(self, *args: Any, config: Config, dataset: Dataset, **kwargs: Any):
        self._current_goal_id: Optional[str] = None

        self._data = torch.load(config.DATA_PATH.format(split=config.DATA_SPLIT))
        # if has multiple images, random sample a image per episodes
        self.should_sample_view = len(list(self._data.keys())[0].split("_")) >= 3
        if self.should_sample_view:
            from collections import Counter
            self.images_counter = Counter(
                ["_".join(k.split("_")[:2]) for k in self._data.keys()]
            )
        
        super().__init__(config=config)

    def _get_example_observation(self):
        object_id = list(self._data.keys())[0]
        obs = self._data[object_id]
        return obs

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,self.config.CLIP_SIZE),
            dtype=np.float32,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_goal(self, id):
        return self._data[id].numpy()

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        scene_short_id = os.path.basename(episode.scene_id).split(".")[0]
        goal_object_id = episode.goal_object_id
        goal_id = f"{scene_short_id}_{goal_object_id}"
        if goal_id == self._current_goal_id:
            return self._current_goal
        
        if self.should_sample_view:
            num_images = self.images_counter[goal_id]
            goal_id += "_{}".format(np.random.choice(num_images))

        self._current_goal_id = goal_id
        self._current_goal = self._get_goal(goal_id)

        return self._current_goal


@registry.register_sensor
class IINImageGoalSensor(Sensor):
    cls_uuid: str = "iin-image-goal"

    def __init__(self, *args: Any, config: Config, dataset: Dataset, **kwargs: Any):
        self._current_goal_id: Optional[str] = None
        self._data_root = config.DATA_PATH.format(split=config.DATA_SPLIT)
        self._data = {"_".join(p.split("-")[:2]):p for p in os.listdir(self._data_root) if p.endswith(".png")}
        
        super().__init__(config=config)

    def _get_example_observation(self):
        object_id = list(self._data.keys())[0]
        image_path = os.path.join(self._data_root, self._data[object_id])
        return np.asarray(Image.open(image_path).convert("RGB"))

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(512,512,3),
            dtype=np.uint8,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_goal(self, id):
        image_path = os.path.join(self._data_root, self._data[id])
        return np.asarray(Image.open(image_path).convert("RGB"))

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        scene_short_id = os.path.basename(episode.scene_id).split(".")[0]
        goal_object_id = episode.goal_object_id
        goal_id = f"{scene_short_id}_{goal_object_id}"
        if goal_id == self._current_goal_id:
            return self._current_goal

        self._current_goal_id = goal_id
        self._current_goal = self._get_goal(goal_id)

        return self._current_goal


@registry.register_sensor
class ObjectDistanceSensor(Sensor):
    cls_uuid: str = "object-distance"

    def __init__(self, *args: Any, config: Config, dataset: Dataset, **kwargs: Any):
        self._current_episode_id: Optional[str] = None
        self._current_value = None

        self._data = {}
        self._angle = {}
        for scene_id in dataset.scene_ids:
            scene = dataset.scene_from_scene_path(scene_id)
            path = config.DATA_PATH.format(split=config.DATA_SPLIT, scene=scene)
            if config.DATA_SPLIT == "val" and not os.path.exists(path):
                logger.warning("no distance file found, use default level 1 (1m~2m)")
                self._data[scene_id] = np.ones((9000, 1, 1), dtype="float16")
            else:
                self._data[scene_id] = np.load(path, mmap_mode="r")

        super().__init__(config=config)
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH
    
    def value_mapping(self, x):
        assert x >= 0
        # if x < 0.5:
        #     return 0 # level 1
        # elif x < 1:
        #     return 1 # level 2
        # elif x < 3:
        #     return 2 # level 3
        # else:
        #     return 3 # level 4
        if x < 10:
            return int(x)
        else:
            return 10

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=0, high=10, shape=(1,), dtype=int
        )

    def _get_value(self, episode):
        scene_id, episode_id = episode.scene_id, int(episode.episode_id)
        obs = self._data[scene_id][episode_id]
        return np.array((self.value_mapping(obs),))

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_value

        self._current_episode_id = episode_uniq_id
        self._current_value = self._get_value(episode)

        return self._current_value


@registry.register_sensor
class AugCachedGoalSensor(CachedGoalSensor):
    cls_uuid: str = "aug-cached-goal"

    def euler_to_quaternion(self, yaw, roll, pitch):
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)

        # calculate quaternion
        w = cp * cy * cr + sp * sy * sr
        x = sp * cy * cr - cp * sy * sr
        y = cp * sy * cr + sp * cy * sr
        z = cp * cy * sr - sp * sy * cr

        return quaternion_from_coeff([x, y, z, w])

    def _get_goal(self, episode):
        """order: left, front, right, back."""
        scene_id, episode_id = episode.scene_id, episode.episode_id
        obs = self._data[scene_id][episode_id]

        if self.config.SINGLE_VIEW and self.config.SAMPLE_GOAL_ANGLE:
            # sample an observation index and offset the default angle
            obs_idx = np.random.choice(4)
        else:
            obs_idx = 0

        yaw = float(self._angle[scene_id][episode_id][obs_idx][0])
        pitch = float(self._angle[scene_id][episode_id][obs_idx][1])

        # set the goal rotation
        # goal_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = self.euler_to_quaternion(yaw, 0, pitch)

        # return the observation embedding
        if self.config.SINGLE_VIEW:
            return obs[obs_idx][None]
        else:
            return obs
