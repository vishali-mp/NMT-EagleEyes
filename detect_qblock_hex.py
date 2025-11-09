# detect_qblock_hex.py
# Detect q-blocks, compute average HEX color for each, save annotated image + CSV.

from ultralytics import YOLO
import cv2, numpy as np, pandas as pd
from pathlib import Path

# --------- CONFIG (edit these paths) ---------
MODEL_PATH = "./runs/detect/yolov8n_qblock_detector/weights/best.pt"   # your trained weights
SOURCE     = "./eagle-eyes-22/test/images"                         # image file OR folder
OUTPUT_DIR = "out_qblocks"                          # where results go
CONF       = 0.15                                  # lower -> more detections
IMG_SIZE   = 960                                   # try 640/960/1280
ROI_SHRINK = 2                                     # shave pixels inside crop before color
ROW_EPS_FRAC = 0.03                                # row grouping tolerance (fraction of image height)
AREA_THRESH_MODE = "auto"                                  # shave a few pixels off the crop to avoid bleed
# ---------------------------------------------


def rgb_to_hex(r, g, b):
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

def median_hex_from_crop(bgr, shrink=2):
    """Robust color: median RGB in crop (optional border shrink)."""
    if bgr is None or bgr.size == 0:
        return "#000000"
    h, w = bgr.shape[:2]
    y1 = max(0, shrink); y2 = max(0, h - shrink)
    x1 = max(0, shrink); x2 = max(0, w - shrink)
    roi = bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else bgr
    if roi.ndim == 2 or roi.shape[2] == 1:
        # grayscale
        g = float(np.median(roi))
        return rgb_to_hex(g, g, g)
    b = float(np.median(roi[:, :, 0]))
    g = float(np.median(roi[:, :, 1]))
    r = float(np.median(roi[:, :, 2]))
    return rgb_to_hex(r, g, b)

def sort_reading_order(boxes, img_h, row_eps_frac=0.03):
    """
    boxes: list of dicts with keys {x1,y1,x2,y2,cx,cy,area}
    Sort by rows (top->bottom) then left->right within row.
    """
    eps = row_eps_frac * img_h
    # Assign a row key based on cy quantization
    rows = {}
    for b in boxes:
        cy = b["cy"]
        # find an existing row whose mean is close
        matched_key = None
        for rk in rows:
            if abs(cy - rk) <= eps:
                matched_key = rk
                break
        if matched_key is None:
            rows[cy] = [b]
        else:
            rows[matched_key].append(b)

    # sort rows by their key (top to bottom)
    ordered = []
    for rk in sorted(rows.keys()):
        row = rows[rk]
        row = sorted(row, key=lambda d: d["cx"])  # left to right
        ordered.extend(row)
    return ordered

def process_image(model, img_path: Path, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[warn] cannot read {img_path}")
        return

    res = model.predict(source=str(img_path), conf=CONF, imgsz=IMG_SIZE, save=False, verbose=False)[0]
    img_h, img_w = img.shape[:2]

    # Collect detections
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w - 1, x2); y2 = min(img_h - 1, y2)
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        area = w * h
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h, "cx": cx, "cy": cy, "area": area, "conf": conf})

    if not dets:
        print(f"[info] no detections in {img_path.name}")
        return

    # Decide large vs small
    if AREA_THRESH_MODE == "auto":
        areas = np.array([d["area"] for d in dets], dtype=float)
        area_thresh = float(np.median(areas))  # robust split
    else:
        area_thresh = float(AREA_THRESH_MODE)

    large = [d for d in dets if d["area"] >= area_thresh]
    small = [d for d in dets if d["area"] < area_thresh]

    # Sort each group in reading order & assign IDs
    large = sort_reading_order(large, img_h, ROW_EPS_FRAC)
    small = sort_reading_order(small, img_h, ROW_EPS_FRAC)

    for i, d in enumerate(large, 1):
        d["id"] = f"q{i}"
        d["kind"] = "large"
    for i, d in enumerate(small, 1):
        d["id"] = f"sq{i}"
        d["kind"] = "small"

    all_boxes = large + small

    # Compute HEX for each detection and draw labels
    out_img = img.copy()
    rows = []
    for d in all_boxes:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        crop = img[y1:y2, x1:x2]
        hex_color = median_hex_from_crop(crop, ROI_SHRINK)
        d["hex"] = hex_color

        color = (0, 255, 0) if d["kind"] == "large" else (255, 0, 0)  # large=green, small=red
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        label = f"{d['id']} {hex_color}"
        cv2.putText(out_img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        rows.append({
            "image": img_path.name,
            "id": d["id"],
            "type": d["kind"],
            "hex": hex_color,
            "conf": round(d["conf"], 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": d["w"], "h": d["h"], "area": d["area"]
        })

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / f"{img_path.stem}_annotated.jpg"
    csv_path     = out_dir / f"{img_path.stem}_qblock_hex.csv"

    cv2.imwrite(str(out_img_path), out_img)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"[saved] {out_img_path}")
    print(f"[saved] {csv_path}")
    print(f"[info] large={len(large)} small={len(small)}  (area_thresh={int(area_thresh)})")

def main():
    root = Path(".").resolve()
    model = YOLO(str(root / MODEL_PATH))

    src = root / SOURCE
    out_dir = root / OUTPUT_DIR
    if src.is_dir():
        imgs = [p for p in src.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff")]
        imgs.sort()
        for p in imgs:
            process_image(model, p, out_dir)
    else:
        process_image(model, src, out_dir)

if __name__ == "__main__":
    main()
