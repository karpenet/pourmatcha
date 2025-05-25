#!/usr/bin/env python
"""
Add (or overwrite) Grounding‑DINO bounding‑box annotations for every RGB frame
in a LeRobot / HF‑datasets split.

This **version uses the Hugging Face `transformers` implementation** of
Grounding‑DINO (`AutoProcessor` + `AutoModelForZeroShotObjectDetection`).  It
works entirely with the public model *IDEA‑Research/grounding-dino‑tiny* (or
any compatible checkpoint) and requires nothing outside the standard HF
stack.

Workflow
--------
1. Download the requested dataset split from the Hub.
2. For each row, feed the image through Grounding‑DINO with the user‑supplied
   *prompt* (list of strings).  The model returns boxes in **absolute pixel
   coordinates** plus labels and scores.
3. Store those boxes as a list‑of‑dicts in a new column called **`bboxes`**.
4. Optionally draw the boxes on the image for quick visual QA.
5. Save the augmented dataset locally and push it (together with a proper
   `meta/info.json` and matching tag) back to the Hub so that LeRobot can load
   it.

Example
-------
```bash
python add_bboxes_with_grounding_dino.py \
  --source-repo   jchun/so100_cleaning_20250524_161038 \
  --target-repo   jchun/so100_with_boxes \
  --split         train \
  --image-column  observation.image \
  --prompt        "a cat,a remote control" \
  --box-thresh    0.4 \
  --text-thresh   0.3 \
  --draw-images
```

Dependencies
------------
* **torch >= 2.1**
* **transformers >= 4.41** (Grounding‑DINO support)
* pillow
* datasets >= 2.18
* tqdm
* huggingface‑hub
"""

from __future__ import annotations

# ── std‑lib ─────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import warnings

# ── third‑party ─────────────────────────────────────────────────────────────
import torch                                  # torch/__init__.py
from datasets import Dataset, load_dataset    # datasets/arrow_dataset.py
from huggingface_hub import HfApi            # huggingface_hub/hf_api.py
from PIL import Image, ImageDraw              # PIL/Image.py, PIL/ImageDraw.py
from tqdm import tqdm                         # tqdm/std.py
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

###############################################################################
# Helper functions                                                            #
###############################################################################

def detect_boxes(
    model,
    processor,
    image: Image.Image,
    prompts: List[str],
    box_thresh: float,
    text_thresh: float,
    device: str,
) -> List[Dict[str, Any]]:
    """Run Grounding‑DINO and return boxes in absolute pixel coordinates."""
    # model expects batched inputs
    inputs = processor(images=image, text=[prompts], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]

    boxes_px = results["boxes"].tolist()          # xyxy, already absolute
    scores   = results["scores"].tolist()
    labels   = results["labels"]                 # list[str]

    out: List[Dict[str, Any]] = []
    for b, s, l in zip(boxes_px, scores, labels):
        x1, y1, x2, y2 = [float(round(v, 2)) for v in b]
        out.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(round(s, 4)),
            "label": l,
        })
    return out


def draw_boxes(image: Image.Image, boxes: List[Dict[str, Any]]) -> Image.Image:
    """Return an RGB image with red rectangles for each box."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([(b["x1"], b["y1"]), (b["x2"], b["y2"])], outline="red", width=3)
    return img

###############################################################################
# Main                                                                        #
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Add Grounding‑DINO boxes (transformers version)")
    # Dataset I/O
    parser.add_argument("--source-repo", required=True, help="Source HF dataset repo id")
    parser.add_argument("--target-repo", required=True, help="Repo id to push augmented dataset")
    parser.add_argument("--split", default="train", help="Split to process (default: train)")
    parser.add_argument("--image-column", default="image", help="Column with PIL images or paths")
    # Grounding‑DINO params
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny", help="Model checkpoint")
    parser.add_argument("--prompt", required=True, help="Comma‑separated object names")
    parser.add_argument("--box-thresh", type=float, default=0.4, help="Box score threshold")
    parser.add_argument("--text-thresh", type=float, default=0.3, help="Text score threshold")
    parser.add_argument("--device", default="cuda", help="torch device (cuda / cpu)")
    # Misc
    parser.add_argument("--draw-images", action="store_true", help="Save debug images with boxes drawn")
    parser.add_argument("--local-out", type=Path, default=Path("./dataset_with_boxes"), help="Local output dir")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    if not prompts:
        raise ValueError("--prompt must contain at least one label string")

    # ------------------------------------------------------------------
    print(f"⏬ Loading {args.source_repo}:{args.split} …")
    ds: Dataset = LeRobotDataset(args.source_repo)
    # ds = LeRobotDataset("jchun/so100_cleaning_20250524_161038", split="train")

    # ------------------------------------------------------------------
    print(f"📦 Loading Grounding‑DINO ({args.model_id}) …")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)
    model.eval()

    out_rows: List[Dict[str, Any]] = []
    draw_dir = args.local_out / "drawn" if args.draw_images else None
    if draw_dir:
        draw_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(list(enumerate(ds)), desc="Annotating"):
        # print(row)
        img_obj = row[args.image_column]
        if isinstance(img_obj, Image.Image):
            img = img_obj
        elif isinstance(img_obj, torch.Tensor):
            # Convert tensor to PIL Image
            img = Image.fromarray((img_obj.cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0))
        else:
            img = Image.open(img_obj)

        boxes = detect_boxes(
            model,
            processor,
            img,
            prompts,
            args.box_thresh,
            args.text_thresh,
            args.device,
        )
        row["bboxes"] = boxes

        if draw_dir is not None:
            drawn = draw_boxes(img, boxes)
            out_path = draw_dir / f"{idx:06d}.jpg"
            drawn.save(out_path)
            row["debug_image"] = str(out_path.relative_to(args.local_out))

        out_rows.append(row)

    # ------------------------------------------------------------------
    print("📝 Building Arrow table …")
    new_ds = Dataset.from_list(out_rows)

    print(f"💾 Saving to {args.local_out}")
    new_ds.save_to_disk(str(args.local_out))

    # ------------------------------------------------------------------
    meta_dir = args.local_out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "name": args.target_repo,
        "description": "Dataset with Grounding‑DINO (transformers) bounding boxes",
        "codebase_version": "0.1.0",
        "creation_time": datetime.now().strftime("%Y-%m-%d"),
        "data_format_version": "1.0.0",
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    # ------------------------------------------------------------------
    print(f"🚀 Pushing to HF Hub → {args.target_repo}")
    api = HfApi()
    api.create_repo(args.target_repo, repo_type="dataset", exist_ok=True)

    new_ds.push_to_hub(args.target_repo, commit_message="Add bounding boxes (Grounding‑DINO transformers)")
    api.upload_folder(
        repo_id=args.target_repo,
        repo_type="dataset",
        folder_path=str(meta_dir),
        path_in_repo="meta",
        commit_message="Add meta/info.json (codebase_version 0.1.0)",
    )
    api.create_tag(args.target_repo, tag="0.1.0", repo_type="dataset")

    print("✅ Done – dataset with boxes is live!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", module="transformers")
    main()
