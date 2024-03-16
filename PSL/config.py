import os
import sys
import math
import shutil
import socket
import subprocess
import warnings
from glob import glob
from datetime import datetime
from typing import List, Optional, Union
from pathlib import Path
from shlex import quote

from habitat.config.default import _C as _HABITAT_CONFIG
from habitat.config.default import Config as CN
from habitat_baselines.config.default import _C as _BASE_CONFIG

import logging
logger = logging.getLogger(__name__)

CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------

# fmt:off
_TASK_CONFIG = _HABITAT_CONFIG.clone()
_TASK_CONFIG.defrost()

_TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 500

_TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30

_TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False

# LEFT
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR = CN()
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.UUID = "rgb_left"
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.TYPE = "HabitatSimRGBSensor"
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.SENSOR_SUBTYPE = "PINHOLE"
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.HFOV = 79
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.WIDTH = 640
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.HEIGHT = 480
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.POSITION = [0.0, 0.88, 0.0]
_TASK_CONFIG.SIMULATOR.RGB_LEFT_SENSOR.ORIENTATION = [0.0, 1 / 2 * math.pi, 0.0]  # Euler angles

# FORWARD
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.UUID = "rgb"
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.SENSOR_SUBTYPE = "PINHOLE"
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV = 79
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 640
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 480
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0.0, 0.88, 0.0]
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler angles

# RIGHT
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR = CN()
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.UUID = "rgb_right"
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.TYPE = "HabitatSimRGBSensor"
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.SENSOR_SUBTYPE = "PINHOLE"
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.HFOV = 79
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.WIDTH = 640
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.HEIGHT = 480
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.POSITION = [0.0, 0.88, 0.0]
_TASK_CONFIG.SIMULATOR.RGB_RIGHT_SENSOR.ORIENTATION = [0.0, 3 / 2 * math.pi, 0.0]  # Euler angles

# BACK
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR = CN()
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.UUID = "rgb_back"
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.TYPE = "HabitatSimRGBSensor"
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.SENSOR_SUBTYPE = "PINHOLE"
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.HFOV = 79
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.WIDTH = 640
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.HEIGHT = 480
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.POSITION = [0.0, 0.88, 0.0]
_TASK_CONFIG.SIMULATOR.RGB_BACK_SENSOR.ORIENTATION = [0.0, math.pi, 0.0]  # Euler angles
# fmt:on

_TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 0.88
_TASK_CONFIG.SIMULATOR.AGENT_0.RADIUS = 0.18
_TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]

_TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

_TASK_CONFIG.TASK.PANORAMIC_IMAGEGOAL_SENSOR = CN()
_TASK_CONFIG.TASK.PANORAMIC_IMAGEGOAL_SENSOR.TYPE = "PanoramicImageGoalSensor"
_TASK_CONFIG.TASK.PANORAMIC_IMAGEGOAL_SENSOR.SAMPLE_GOAL_ANGLE = False

_TASK_CONFIG.TASK.ANGLE_TO_GOAL = CN()
_TASK_CONFIG.TASK.ANGLE_TO_GOAL.TYPE = "AngleToGoal"

_TASK_CONFIG.TASK.ANGLE_SUCCESS = CN()
_TASK_CONFIG.TASK.ANGLE_SUCCESS.TYPE = "AngleSuccess"
_TASK_CONFIG.TASK.ANGLE_SUCCESS.SUCCESS_ANGLE = 25.0

# OBJECTGOAL_PROMPT_SENSOR
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR = CN()
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR.TYPE = "ObjectGoalPromptSensor"
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR.HAS_ATTRIBUTE = False
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR.ATTRIBUTE_TYPE = "intrinsic_attributes"
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR.PARSE = False
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR.PROMPT = "{cat}"
_TASK_CONFIG.TASK.OBJECTGOAL_PROMPT_SENSOR.CLIP_MODEL = "RN50"

# IMAGE_GOAL_SENSOR
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR.TYPE = "ImageGoalSensorV2"
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR.CHANNELS = ["rgb"]
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR.SAMPLE_TYPE = "uniform"
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR.DATA_SPLIT = "train"
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR.SINGLE_VIEW = True
_TASK_CONFIG.TASK.IMAGE_GOAL_SENSOR.SAMPLE_GOAL_ANGLE = False

# CACHED_GOAL_SENSOR
_TASK_CONFIG.TASK.CACHED_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.TYPE = "CachedGoalSensor"
_TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.DATA_PATH = (
    "data/goal_datasets/imagenav/hm3d/v2/{split}/content/{scene}.npy"
)
_TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.DATA_SPLIT = "train"
_TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.SINGLE_VIEW = True
_TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.SAMPLE_GOAL_ANGLE = False

# AUG_CACHED_GOAL_SENSOR
_TASK_CONFIG.TASK.AUG_CACHED_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.AUG_CACHED_GOAL_SENSOR.TYPE = "AugCachedGoalSensor"
_TASK_CONFIG.TASK.AUG_CACHED_GOAL_SENSOR.DATA_PATH = (
    "data/goal_datasets/imagenav/hm3d/v2/{split}/content/{scene}.npy"
)
_TASK_CONFIG.TASK.AUG_CACHED_GOAL_SENSOR.DATA_SPLIT = "train"
_TASK_CONFIG.TASK.AUG_CACHED_GOAL_SENSOR.SINGLE_VIEW = True
_TASK_CONFIG.TASK.AUG_CACHED_GOAL_SENSOR.SAMPLE_GOAL_ANGLE = False

# IIN CACHED_GOAL_SENSOR
_TASK_CONFIG.TASK.IIN_CACHED_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.IIN_CACHED_GOAL_SENSOR.TYPE = "IINCachedGoalSensor"
_TASK_CONFIG.TASK.IIN_CACHED_GOAL_SENSOR.DATA_PATH = (
    "data/goal_datasets/instance_imagenav_hm3d_v3/{split}.pth"
)
_TASK_CONFIG.TASK.IIN_CACHED_GOAL_SENSOR.DATA_SPLIT = "val"
_TASK_CONFIG.TASK.IIN_CACHED_GOAL_SENSOR.CLIP_SIZE = 1024

# IIN IMAGE_GOAL_SENSOR
_TASK_CONFIG.TASK.IIN_IMAGE_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.IIN_IMAGE_GOAL_SENSOR.TYPE = "IINImageGoalSensor"
_TASK_CONFIG.TASK.IIN_IMAGE_GOAL_SENSOR.DATA_PATH = (
    "images/{split}/"
)
_TASK_CONFIG.TASK.IIN_IMAGE_GOAL_SENSOR.DATA_SPLIT = "val"

# COMPL ATTR CACHED GOAL SENSOR
_TASK_CONFIG.TASK.COMPL_ATTR_CACHED_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.COMPL_ATTR_CACHED_GOAL_SENSOR.TYPE = "ComplAttrCachedGoalSensor"
_TASK_CONFIG.TASK.COMPL_ATTR_CACHED_GOAL_SENSOR.DATA_PATH = (
    "data/goal_datasets/instance_imagenav_hm3d_v3/{split}.complementary.pth"
)
_TASK_CONFIG.TASK.COMPL_ATTR_CACHED_GOAL_SENSOR.DATA_SPLIT = "val"
_TASK_CONFIG.TASK.COMPL_ATTR_CACHED_GOAL_SENSOR.CLIP_SIZE = 1024

# QUERIED IMAGE GOAL SENSOR
_TASK_CONFIG.TASK.QUERIED_IMAGE_GOAL_SENSOR = CN()
_TASK_CONFIG.TASK.QUERIED_IMAGE_GOAL_SENSOR.TYPE = "QueriedImageGoalSensor"
_TASK_CONFIG.TASK.QUERIED_IMAGE_GOAL_SENSOR.DATA_PATH = (
    "data/goal_datasets/instance_imagenav_hm3d_v3/{split}.queried_image_goal.pth"
)
_TASK_CONFIG.TASK.QUERIED_IMAGE_GOAL_SENSOR.DATA_SPLIT = "val"
_TASK_CONFIG.TASK.QUERIED_IMAGE_GOAL_SENSOR.CLIP_SIZE = 1024

# OBJECT DISTANCE SENSOR
_TASK_CONFIG.TASK.OBJECT_DISTANCE_SENSOR = CN()
_TASK_CONFIG.TASK.OBJECT_DISTANCE_SENSOR.TYPE = "ObjectDistanceSensor"
_TASK_CONFIG.TASK.OBJECT_DISTANCE_SENSOR.DATA_PATH = (
    "data/goal_datasets/imagenav/hm3d/v2/{split}/content/{scene}.distance.npy"
)
_TASK_CONFIG.TASK.OBJECT_DISTANCE_SENSOR.DATA_SPLIT = "train"
_TASK_CONFIG.TASK.OBJECT_DISTANCE_SENSOR.REPRESENTATION = "category"

# OBJECTGOAL_KSHOT_IMAGE_PROMPT_SENSOR
_TASK_CONFIG.TASK.OBJECTGOAL_KSHOT_IMAGE_PROMPT_SENSOR = CN()
_TASK_CONFIG.TASK.OBJECTGOAL_KSHOT_IMAGE_PROMPT_SENSOR.TYPE = (
    "ObjectGoalKShotImagePromptSensor"
)
_TASK_CONFIG.TASK.OBJECTGOAL_KSHOT_IMAGE_PROMPT_SENSOR.SHOTS = 1
_TASK_CONFIG.TASK.OBJECTGOAL_KSHOT_IMAGE_PROMPT_SENSOR.DATA_PATH = (
    "data/kshots_datasets/hm3d/object_goal_kshot_images.npy"
)
_TASK_CONFIG.TASK.OBJECTGOAL_KSHOT_IMAGE_PROMPT_SENSOR.CLIP_SIZE = 1024

# SIMPLE_REWARD
_TASK_CONFIG.TASK.SIMPLE_REWARD = CN()
_TASK_CONFIG.TASK.SIMPLE_REWARD.TYPE = "SimpleReward"
_TASK_CONFIG.TASK.SIMPLE_REWARD.SUCCESS_REWARD = 2.5
_TASK_CONFIG.TASK.SIMPLE_REWARD.ANGLE_SUCCESS_REWARD = 0.0
_TASK_CONFIG.TASK.SIMPLE_REWARD.SLACK_PENALTY = -0.01
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD = False
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_REWARD = False
_TASK_CONFIG.TASK.SIMPLE_REWARD.ATG_REWARD_DISTANCE = 1.0
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_FIX = False

_TASK_CONFIG.TASK.AGENT_POSITION = CN()
_TASK_CONFIG.TASK.AGENT_POSITION.TYPE = "AgentPosition"

_TASK_CONFIG.TASK.AGENT_ROTATION = CN()
_TASK_CONFIG.TASK.AGENT_ROTATION.TYPE = "AgentRotation"

_TASK_CONFIG.TASK.SENSORS = ["PANORAMIC_IMAGEGOAL_SENSOR"]
_TASK_CONFIG.TASK.MEASUREMENTS = [
    "DISTANCE_TO_GOAL",
    "SUCCESS",
    "SPL",
    "SOFT_SPL",
    "SIMPLE_REWARD",
]


def get_task_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _TASK_CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

_CONFIG = _BASE_CONFIG.clone()
_CONFIG.defrost()

_CONFIG.VERBOSE = True

_CONFIG.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"

_CONFIG.TRAINER_NAME = "zson-ppo"
_CONFIG.ENV_NAME = "SimpleRLEnv"
_CONFIG.SENSORS = ["RGB_SENSOR"]

_CONFIG.VIDEO_OPTION = []
_CONFIG.VIDEO_DIR = "data/video"
_CONFIG.TENSORBOARD_DIR = "data/tensorboard"
_CONFIG.EVAL_CKPT_PATH_DIR = "data/checkpoints"
_CONFIG.EVAL_CKPT_START_ID = -1
_CONFIG.CHECKPOINT_FOLDER = "data/checkpoints"
_CONFIG.LOG_FILE = "data/train.log"
_CONFIG.DETERMINISTIC = False
_CONFIG.SAVE_LAST_OBS = False

_CONFIG.NUM_ENVIRONMENTS = 12
_CONFIG.LOG_INTERVAL = 10
_CONFIG.NUM_CHECKPOINTS = 100
_CONFIG.NUM_UPDATES = -1
_CONFIG.TOTAL_NUM_STEPS = 250e6
_CONFIG.FORCE_TORCH_SINGLE_THREADED = True

_CONFIG.RUN_TYPE = None

_CONFIG.EVAL.SPLIT = "val"
_CONFIG.EVAL.USE_CKPT_CONFIG = False
_CONFIG.EVAL.evaluation_meta_file = "data/tensorboard/eval_metrics.json"
_CONFIG.EVAL.avg_eval_metrics = "data/tensorboard/avg_eval_metrics.json"
_CONFIG.EVAL.episodes_eval_data = False
_CONFIG.RL.REWARD_MEASURE = "simple_reward"
_CONFIG.RL.SUCCESS_MEASURE = "success"

_CONFIG.RL.POLICY.name = "ZSONPolicy"
_CONFIG.RL.POLICY.backbone = "resnet50"
_CONFIG.RL.POLICY.baseplanes = 32
_CONFIG.RL.POLICY.hidden_size = 512
_CONFIG.RL.POLICY.rnn_type = "GRU"
_CONFIG.RL.POLICY.rnn_num_layers = 2
_CONFIG.RL.POLICY.use_data_aug = True
_CONFIG.RL.POLICY.pretrained_encoder = None
_CONFIG.RL.POLICY.CLIP_MODEL = "RN50"
_CONFIG.RL.POLICY.OBS_CLIP_MODEL = "RN50"
_CONFIG.RL.POLICY.use_clip_obs_encoder = False
_CONFIG.RL.POLICY.train_goal_encoder = False
_CONFIG.RL.POLICY.enable_fg_midfusion = False
_CONFIG.RL.POLICY.enable_sm_midfusion = False
_CONFIG.RL.POLICY.film_reduction = "none"
_CONFIG.RL.POLICY.use_layout_encoder = False
_CONFIG.RL.POLICY.use_clip_corr_mapping = False
_CONFIG.RL.POLICY.corr_hidden_size = 512
_CONFIG.RL.POLICY.gradient_image = False
_CONFIG.RL.POLICY.cam_visual = False

_CONFIG.RL.PPO.clip_param = 0.2
_CONFIG.RL.PPO.ppo_epoch = 4
_CONFIG.RL.PPO.num_mini_batch = 2
_CONFIG.RL.PPO.value_loss_coef = 0.5
_CONFIG.RL.PPO.entropy_coef = 0.01
_CONFIG.RL.PPO.lr = 1.25e-4
_CONFIG.RL.PPO.wd = 1e-6
_CONFIG.RL.PPO.eps = 1e-5
_CONFIG.RL.PPO.max_grad_norm = 0.5
_CONFIG.RL.PPO.num_steps = 64
_CONFIG.RL.PPO.use_gae = True
_CONFIG.RL.PPO.use_linear_lr_decay = False
_CONFIG.RL.PPO.use_linear_clip_decay = False
_CONFIG.RL.PPO.gamma = 0.99
_CONFIG.RL.PPO.tau = 0.95
_CONFIG.RL.PPO.reward_window_size = 50
_CONFIG.RL.PPO.use_normalized_advantage = False
_CONFIG.RL.PPO.hidden_size = 512
_CONFIG.RL.PPO.use_double_buffered_sampler = False

_CONFIG.RL.DDPPO.sync_frac = 0.6
_CONFIG.RL.DDPPO.distrib_backend = "NCCL"


def get_random_rundir(exp_dir: Path, prefix: str = 'run', suffix: str = ''):
    if exp_dir.exists():
        runs = glob(str(exp_dir / '*_run*'))
        num_runs = len([r for r in runs if (prefix + '_') in r])
    else:
        num_runs = 0
    dt = datetime.now().strftime('%Y%m%d-%H%M%S')
    rundir = prefix + '_' + 'run{}'.format(num_runs) + '_' + suffix
    return (exp_dir / rundir)


def pack_code(run_dir: str):
    run_dir = Path(run_dir) / 'code'
    if not run_dir.exists():
        run_dir.mkdir()
    if os.path.isdir(".git"):
        HEAD_commit_id = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True
        )
        tar_name = f'code_{HEAD_commit_id.stdout[:-1]}.tar.gz'
        subprocess.run(
            ['git', 'archive', '-o', str(run_dir/tar_name), 'HEAD'],
            check=True,
        )
        diff_process = subprocess.run(
            ['git', 'diff', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        if diff_process.stdout:
            logger.warning('Working tree is dirty. Patch:\n%s', diff_process.stdout)
            with (run_dir / 'dirty.patch').open('w') as f:
                f.write(diff_process.stdout)
    else:
        logger.warning('.git does not exist in current dir')


def save_sh(run_dir, run_type):
    with open(run_dir / 'code/run_{}_{}.sh'.format(run_type, socket.gethostname()), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('mkdir -p {}\n'.format(run_dir / 'code/unpack'))
        f.write('tar -C {} -xzvf {}\n'.format(run_dir / 'code/unpack', run_dir / 'code/code_*.tar.gz'))
        f.write('cd {}\n'.format(run_dir / 'code/unpack'))
        f.write('patch -p1 < ../dirty.patch\n')
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('cp -r -f {} {}\n'.format(run_dir / 'code/unpack/*', quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')


def save_config(run_dir, config, type):
    if config is not None:
        F = open(os.path.join(run_dir, 'config_of_{}.yaml'.format(type)), 'w')
        F.write(str(config))
        F.close()


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    run_type: Optional[str] = None,
    model_dir: Optional[str] = None,
    overwrite: Optional[bool] = False,
    world_rank: Optional[int] = 0,
    note: Optional[str] = None,
    debug: Optional[bool] = False,
    seed: Optional[int] = 0,
) -> CN:
    config = _CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)

    # write to config yaml
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.SEED = seed

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    if config.NUM_PROCESSES != -1:
        warnings.warn(
            "NUM_PROCESSES is deprecated and will be removed in a future version."
            "  Use NUM_ENVIRONMENTS instead."
            "  Overwriting NUM_ENVIRONMENTS with NUM_PROCESSES for backwards compatibility."
        )

        config.NUM_ENVIRONMENTS = config.NUM_PROCESSES

    config.CHECKPOINT_FOLDER = os.path.join(model_dir, 'ckpts')
    if 'EVAL_CKPT_PATH_DIR' not in opts:
        config.EVAL_CKPT_PATH_DIR = os.path.join(model_dir, 'ckpts')
    config.TENSORBOARD_DIR = os.path.join(model_dir, 'tb')
    config.VIDEO_DIR = os.path.join(model_dir, "video")

    if debug:
        if "gibson" in config.TASK_CONFIG.DATASET.DATA_PATH:
            scene = ['Adrian']
        elif "mp3d" in config.TASK_CONFIG.DATASET.DATA_PATH:
            scene = ['pRbA3pwrgk9']
        elif "hm3d" in config.TASK_CONFIG.DATASET.DATA_PATH:
            # scene = ['00744-1S7LAXRdDqK', '00592-CthA7sQNTPK']
            scene = ['00720-8B43pG641ff', '00262-1xGrZPxG1Hz']
        else:
            scene = ['*']

        logger.warning('Debugging with only 2 scene!')
        config.TASK_CONFIG.DATASET.CONTENT_SCENES.clear()
        config.TASK_CONFIG.DATASET.CONTENT_SCENES.extend(scene)
        config.NUM_ENVIRONMENTS = 2
        config.LOG_INTERVAL = 1

    if world_rank == 0:
        if overwrite:
            logger.warning('Warning! overwrite is specified!\nCurrent model dir will be removed!')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=True)

        run_dir = get_random_rundir(exp_dir=Path(model_dir), prefix=run_type, suffix=note)
        
        config.LOG_FILE = os.path.join(run_dir, 'log.txt')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)

        pack_code(run_dir)
        save_sh(run_dir, run_type)
        save_config(run_dir, config, run_type)

    config.freeze()
    return config
