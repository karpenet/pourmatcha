cd lerobot
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
cd ..
