CUDA_VISIBLE_DEVICES=0 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python run.py \
    --exp-config configs/experiments/objectnav_mp3d.yaml \
    --run-type eval --model-dir exps/eval/PSL-objectnav/ \
    --note seed_200 --seed 200 \
    EVAL_CKPT_PATH_DIR data/models/PSL_Instancenav.pth \
    EVAL.SPLIT val \
    EVAL.episodes_eval_data True \
    NUM_ENVIRONMENTS 11 \
    TASK_CONFIG.TASK.SENSORS "['QUERIED_IMAGE_GOAL_SENSOR']" \
    TASK_CONFIG.TASK.MEASUREMENTS "['DISTANCE_TO_GOAL','SUCCESS','SPL','SOFT_SPL','AGENT_ROTATION','AGENT_POSITION']" \
    TASK_CONFIG.DATASET.DATA_PATH data/datasets/objectnav/objectnav_hm3d_v1/val/val.json.gz \
    TASK_CONFIG.TASK.POSSIBLE_ACTIONS "['STOP','MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','LOOK_DOWN','LOOK_UP']" \
    TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG v1 \
    TASK_CONFIG.TASK.QUERIED_IMAGE_GOAL_SENSOR.DATA_PATH data/goal_datasets/objectnav/val.queried_image_goal.imagenav_v2.pth \
    RL.REWARD_MEASURE distance_to_goal \
    RL.POLICY.use_clip_obs_encoder True \
    RL.POLICY.use_layout_encoder True \
    RL.POLICY.use_clip_corr_mapping True \
    RL.POLICY.CLIP_MODEL RN50