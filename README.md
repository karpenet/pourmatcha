# PourMatcha

## Data Collection
1. Navigate to the LeRobot directory:
   ```bash
   cd lerobot
   ```
2. Follow the instructions in the LeRobot README to create a new environment.

## Training and Inference
1. Navigate to the Isaac-GR00T directory:
   ```bash
   cd Isaac-GR00T
   ```
2. Activate the GR00T environment:
   ```bash
   # Activate the groot environment
   ```


## Teleop runbook

```

ls /dev/ | grep ACM 
to check ports, apply ports sequentially:
jason leader: /dev/ttyACM0
jason follower: /dev/ttyACM1
cv leader: /dev/ttyACM2
cv follower: /dev/ttyACM3
```

```
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/ttyACM2
sudo chmod 666 /dev/ttyACM3


conda activate tinyenv

python lerobot/scripts/control_robot.py   --robot.type=so100   --robot.cameras='{}'   --control.type=calibrate   --control.arms='["main_follower"]'

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate

python lerobot/scripts/control_robot.py --robot.type=so100 --control.type=teleoperate
```

## Cameras

```
Order of connection:
webcam: 0
main: 2
cv: 4

python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

## Collect data

huggingface-cli login

```
REPO_ID="jchun/so100_cleaning_$(date +%Y%m%d_%H%M%S)"
echo "Save to: ${REPO_ID}"
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.single_task="Clean the area in front of the robot" \
  --control.fps=30 \
  --control.repo_id=${REPO_ID} \
  --control.tags='["cleaning"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=120 \
  --control.reset_time_s=30 \
  --control.num_episodes=100 \
  --control.push_to_hub=true

sh record.sh

python lerobot/scripts/visualize_dataset.py \
  --repo-id jchun/so100_cleaning_<>

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=jchun/so100_cleaning_<> \
  --control.episode=0
  
```

## inference

```
# gr00t
rm -r  ~/.cache/huggingface/hub/models--nahidalam--nvidia-gr00t/
conda activate apple_pie

python scripts/inference_service.py --server \
    --model_path nahidalam/so100_pickplace_small_20250323_120056 \
    --embodiment_tag new_embodiment \
    --data_config bimanual_so100 \
    --denoising_steps 4

python scripts/inference_service.py --server \
    --model_path nahidalam/so100_pickplace_small_20250323_120056-7k \
    --embodiment_tag new_embodiment \
    --data_config bimanual_so100 \
    --denoising_steps 4
    

python getting_started/examples/eval_gr00t_so100.py \
 --host 0.0.0.0 \
 --port 5555 \
 --action_horizon 16

 # ACT
conda activate tinyenv
REPO_ID="jchun/eval_act_bimual_so100_$(date +%Y%m%d_%H%M%S)"
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick cakes from the black tray and place them in the left bowl. Pick ice cream cones from the black tray and place them in the right bowl." \
  --control.repo_id=$REPO_ID \
  --control.num_episodes=10 \
  --control.warmup_time_s=2 \
  --control.episode_time_s=120 \
  --control.reset_time_s=60 \
  --control.push_to_hub=true \
  --control.policy.path=/home/jasonx/.cache/huggingface/hub/models--nahidalam--lerobot_act_so100_pickplace_small_20250322_231547/snapshots/a0d415a1c39394f6a1eb3f4f0fdb60805cde9476/pretrained_model/
```
