# PourMatcha

## Data Collection
1. Navigate to the LeRobot directory:
   ```bash
   cd lerobot
   ```
2. Follow the instructions in the LeRobot README to create a new environment.

### Vizualize Data
```bash
   python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so100_test \
  --local-files-only 1
```


### Install Git LFS
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

### Pushing Data to Hub
```bash
    huggingface-cli repo create your-dataset-name --type dataset
```

## Training and Inference
1. Navigate to the Isaac-GR00T directory:
   ```bash
   cd Isaac-GR00T
   ```
2. Activate the GR00T environment:
   ```bash
   # Activate the groot environment
   ```
