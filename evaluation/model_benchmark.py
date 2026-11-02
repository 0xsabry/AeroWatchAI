"""AeroWatch AI — reproducible model-quality benchmark.

Computes *real* detection metrics (not single-IoU estimates) for any
checkpoint against the project's annotated test artifacts:

- Per-class and overall precision / recall at a stated operating point
- mAP50 and mAP50-95 (COCO-style, 101-point interpolated AP, greedy
  confidence-ordered matching) — computed from raw predictions, not from
  Ultralytics' training-time validation
- Confusion matrix at IoU 0.50 (wrong-class matches, background FP/FN)
- Confidence calibration: reliability bins + Expected Calibration Error
- Recall stratified by ground-truth box size (small / medium / large)
- Duplicate-detection count (same-class prediction pairs with IoU > 0.7)
- CPU latency (warm) and process peak RSS for checkpoint loading/inference

Dataset: tests/artifacts/images (30 annotated images, 38 GT boxes:
Drone 16, Aircraft 16, Bird 4, Balloon 2) + tests/artifacts/negative_cases
(5 images, 0 GT — every detection there is a false positive).

LIMITATIONS (also printed into every report):
- n=38 GT is small; Balloon (n=2) and Bird (n=4) metrics are indicative
  only, with wide uncertainty. Treat differences < ~5 points cautiously.
- The training set is not in this repository, so train/test leakage
  against the *training* data cannot be ruled out from here; artifact
  filenames show some images originate from source datasets' own
  "train" splits (see manifest.json provenance).
- All timing here is CPU (this environment has no GPU). RTX 3050 figures
  quoted anywhere come from the repo's own earlier artifacts, not from
  this benchmark.

Class mapping: checkpoints that declare the canonical 4 classes are used
directly. COCO checkpoints are evaluated through the same mapping the app
uses (COCO airplane->Aircraft, bird->Bird); classes COCO cannot express
(Drone, Balloon) score zero — that is the honest result for that model on
this task, not a harness error.

Usage:
    python evaluation/model_benchmark.py weights/aerowatch_v7_balanced.pt \
        [--imgsz 640] [--nms-iou 0.7] [--op-conf 0.25] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_manager import ModelManager  # noqa: E402

CANONICAL = ["drone", "airplane", "bird", "balloon"]
DISPLAY = {"drone": "Drone", "airplane": "Aircraft", "bird": "Bird", "balloon": "Balloon"}
IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]
OP_IOU = 0.5


def load_gt() -> list[dict]:
    """Ground truth from the artifact label files (YOLO normalised xywh)."""
    from PIL import Image

    items = []
    img_dir = PROJECT_ROOT / "tests" / "artifacts" / "images"
    for label in sorted(img_dir.glob("*.txt")):
        img = label.with_suffix(".jpg")
        if not img.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        gts = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            gts.append(
                {
                    "cls": CANONICAL[cid],
                    "box": [
                        (xc - bw / 2) * w,
                        (yc - bh / 2) * h,
                        (xc + bw / 2) * w,
                        (yc + bh / 2) * h,
                    ],
                    "area_rel": bw * bh,
                }
            )
        items.append({"image": str(img), "w": w, "h": h, "gt": gts})
    neg_dir = PROJECT_ROOT / "tests" / "artifacts" / "negative_cases"
    for img in sorted(neg_dir.glob("*.jpg")):
        with Image.open(img) as im:
            w, h = im.size
        items.append({"image": str(img), "w": w, "h": h, "gt": []})
    return items


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    total = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(mrec, t, side="left")
        total += mpre[idx] if idx < len(mpre) else 0.0
    return float(total / 101.0)


def evaluate(preds_by_image: list[list[dict]], dataset: list[dict]) -> dict:
    """preds: per-image list of {cls (display name), conf, box} (any conf)."""
    # ---- AP / mAP across IoU thresholds ---------------------------------
    ap_per_class: dict[str, dict[str, float]] = {}
    for cls in CANONICAL:
        gt_total = sum(1 for d in dataset for g in d["gt"] if g["cls"] == cls)
        per_iou = {}
        for thr in IOU_THRESHOLDS:
            flat = []
            for img_idx, preds in enumerate(preds_by_image):
                for p in preds:
                    if p["cls"] == cls:
                        flat.append((p["conf"], img_idx, p["box"]))
            flat.sort(key=lambda t: -t[0])
            matched: dict[int, set] = {}
            tp = np.zeros(len(flat))
            fp = np.zeros(len(flat))
            for k, (_conf, img_idx, box) in enumerate(flat):
                best_iou, best_gi = 0.0, -1
                for gi, g in enumerate(dataset[img_idx]["gt"]):
                    if g["cls"] != cls or gi in matched.get(img_idx, set()):
                        continue
                    v = iou(box, g["box"])
                    if v > best_iou:
                        best_iou, best_gi = v, gi
                if best_iou >= thr and best_gi >= 0:
                    tp[k] = 1
                    matched.setdefault(img_idx, set()).add(best_gi)
                else:
                    fp[k] = 1
            if gt_total == 0 or len(flat) == 0:
                per_iou[f"{thr:.2f}"] = 0.0
                continue
            tp_c = np.cumsum(tp)
            fp_c = np.cumsum(fp)
            recalls = tp_c / gt_total
            precisions = tp_c / np.maximum(1, tp_c + fp_c)
            per_iou[f"{thr:.2f}"] = average_precision(recalls, precisions)
        ap_per_class[cls] = per_iou
    map50 = float(np.mean([ap_per_class[c]["0.50"] for c in CANONICAL]))
    map5095 = float(
        np.mean([np.mean(list(ap_per_class[c].values())) for c in CANONICAL])
    )

    return {"ap_per_class": ap_per_class, "map50": map50, "map50_95": map5095}


def operating_point(
    preds_by_image: list[list[dict]], dataset: list[dict], op_conf: float
) -> dict:
    """P/R, confusion, calibration, size strata, errors at one threshold."""
    names = [DISPLAY[c] for c in CANONICAL]
    confusion = {p: {g: 0 for g in names + ["background"]} for p in names}
    fn_background = {g: 0 for g in names}
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0} for c in CANONICAL}
    size_recall = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    calib_bins = {i: [0, 0] for i in range(10)}  # bin -> [correct, total]
    duplicates = 0
    errors = {"false_positives": [], "false_negatives": [], "wrong_class": []}

    for img_idx, d in enumerate(dataset):
        preds = [p for p in preds_by_image[img_idx] if p["conf"] >= op_conf]
        preds.sort(key=lambda p: -p["conf"])
        # duplicates: same-class pairs overlapping heavily
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if preds[i]["cls"] == preds[j]["cls"] and iou(preds[i]["box"], preds[j]["box"]) > 0.7:
                    duplicates += 1
        matched_gt: set[int] = set()
        for p in preds:
            # best GT of ANY class (for wrong-class confusion)
            best_any, best_any_gi, best_any_cls = 0.0, -1, None
            best_same, best_same_gi = 0.0, -1
            for gi, g in enumerate(d["gt"]):
                if gi in matched_gt:
                    continue
                v = iou(p["box"], g["box"])
                if v > best_any:
                    best_any, best_any_gi, best_any_cls = v, gi, g["cls"]
                if g["cls"] == p["cls"] and v > best_same:
                    best_same, best_same_gi = v, gi
            bin_idx = min(9, int(p["conf"] * 10))
            if best_same >= OP_IOU:
                matched_gt.add(best_same_gi)
                per_class[p["cls"]]["tp"] += 1
                confusion[DISPLAY[p["cls"]]][DISPLAY[p["cls"]]] += 1
                calib_bins[bin_idx][0] += 1
                calib_bins[bin_idx][1] += 1
            elif best_any >= OP_IOU and best_any_cls is not None:
                matched_gt.add(best_any_gi)
                per_class[p["cls"]]["fp"] += 1
                per_class[best_any_cls]["fn"] += 1
                confusion[DISPLAY[p["cls"]]][DISPLAY[best_any_cls]] += 1
                calib_bins[bin_idx][1] += 1
                errors["wrong_class"].append(
                    {"image": Path(d["image"]).name, "pred": DISPLAY[p["cls"]],
                     "actual": DISPLAY[best_any_cls], "conf": round(p["conf"], 3),
                     "iou": round(best_any, 2)}
                )
            else:
                per_class[p["cls"]]["fp"] += 1
                confusion[DISPLAY[p["cls"]]]["background"] += 1
                calib_bins[bin_idx][1] += 1
                errors["false_positives"].append(
                    {"image": Path(d["image"]).name, "pred": DISPLAY[p["cls"]],
                     "conf": round(p["conf"], 3),
                     "negative_image": len(d["gt"]) == 0}
                )
        for gi, g in enumerate(d["gt"]):
            per_class[g["cls"]]["gt"] += 1
            stratum = "small" if g["area_rel"] < 0.01 else ("medium" if g["area_rel"] < 0.09 else "large")
            size_recall[stratum][1] += 1
            if gi in matched_gt:
                size_recall[stratum][0] += 1
            else:
                fn_background[DISPLAY[g["cls"]]] += 1
                errors["false_negatives"].append(
                    {"image": Path(d["image"]).name, "actual": DISPLAY[g["cls"]],
                     "area_rel": round(g["area_rel"], 4), "stratum": stratum}
                )

    for c in CANONICAL:
        pc = per_class[c]
        pc["precision"] = pc["tp"] / (pc["tp"] + pc["fp"]) if (pc["tp"] + pc["fp"]) else 0.0
        pc["recall"] = pc["tp"] / pc["gt"] if pc["gt"] else 0.0
        pc["f1"] = (
            2 * pc["precision"] * pc["recall"] / (pc["precision"] + pc["recall"])
            if (pc["precision"] + pc["recall"])
            else 0.0
        )
    tp = sum(per_class[c]["tp"] for c in CANONICAL)
    fp = sum(per_class[c]["fp"] for c in CANONICAL)
    fn = sum(per_class[c]["fn"] for c in CANONICAL) + sum(fn_background.values())
    gt_all = sum(per_class[c]["gt"] for c in CANONICAL)
    overall = {
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / gt_all if gt_all else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "gt": gt_all,
    }
    overall["f1"] = (
        2 * overall["precision"] * overall["recall"] / (overall["precision"] + overall["recall"])
        if (overall["precision"] + overall["recall"])
        else 0.0
    )
    # Expected Calibration Error over prediction bins
    ece, calib = 0.0, []
    total_preds = sum(v[1] for v in calib_bins.values())
    for i in range(10):
        correct, total = calib_bins[i]
        if total:
            acc = correct / total
            ece += (total / max(1, total_preds)) * abs(acc - (i + 0.5) / 10)
            calib.append({"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": total,
                          "empirical_precision": round(acc, 3)})
    return {
        "op_conf": op_conf,
        "overall": overall,
        "per_class": {DISPLAY[c]: per_class[c] for c in CANONICAL},
        "confusion_pred_x_gt": confusion,
        "fn_to_background": fn_background,
        "size_strata_recall": {
            k: {"matched": v[0], "gt": v[1],
                "recall": (v[0] / v[1]) if v[1] else None}
            for k, v in size_recall.items()
        },
        "calibration": {"ece": round(ece, 4), "bins": calib},
        "duplicate_pairs": duplicates,
        "errors": errors,
    }


def pr_sweep(
    preds_by_image: list[list[dict]], dataset: list[dict]
) -> list[dict]:
    rows = []
    for conf in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
        op = operating_point(preds_by_image, dataset, conf)
        rows.append({"conf": conf, "precision": round(op["overall"]["precision"], 3),
                     "recall": round(op["overall"]["recall"], 3),
                     "f1": round(op["overall"]["f1"], 3),
                     "fp": op["overall"]["fp"], "fn": op["overall"]["fn"]})
    return rows


def run_benchmark(
    weights: str,
    imgsz: int = 640,
    nms_iou: float = 0.7,
    op_conf: float = 0.25,
    canonical_dedup: bool = False,
) -> dict:
    dataset = load_gt()
    mm = ModelManager()
    t0 = time.perf_counter()
    mm.load_model(weights, device="cpu")
    load_s = time.perf_counter() - t0

    preds_by_image: list[list[dict]] = []
    latencies = []
    for idx, d in enumerate(dataset):
        img = np.zeros((d["h"], d["w"], 3), dtype=np.uint8)
        from PIL import Image

        img = np.asarray(Image.open(d["image"]).convert("RGB"))[:, :, ::-1]
        start = time.perf_counter()
        res = mm.model.predict(
            img, conf=0.001, iou=nms_iou, imgsz=imgsz, device="cpu", verbose=False
        )[0]
        latencies.append(time.perf_counter() - start)
        preds = []
        if res.boxes is not None:
            for b in res.boxes:
                mapped = mm.map_detection_label(
                    int(b.cls.item()), float(b.conf.item()), detection_threshold=0.0
                )
                if mapped is None or mapped[0] is None:
                    continue
                canon = mapped[0]  # internal canonical name from ModelManager
                if canon not in CANONICAL:
                    continue
                preds.append({"cls": canon, "conf": float(b.conf.item()),
                              "box": [float(v) for v in b.xyxy[0].tolist()]})
        if canonical_dedup:
            from ai_safety import dedupe_canonical

            as_filter_dicts = [
                {"internal_name": p["cls"], "conf": p["conf"], "box": p["box"]}
                for p in preds
            ]
            kept, _suppressed = dedupe_canonical(as_filter_dicts)
            preds = [
                {"cls": k["internal_name"], "conf": k["conf"], "box": k["box"]}
                for k in kept
            ]
        preds_by_image.append(preds)

    warm = latencies[3:] or latencies
    result = {
        "weights": str(weights),
        "checkpoint_sha256": mm.checkpoint_sha256,
        "declared_classes": getattr(mm, "raw_names", None)
        or (mm.model.names if mm.model is not None else None),
        "imgsz": imgsz,
        "nms_iou": nms_iou,
        "canonical_dedup": canonical_dedup,
        "dataset": {
            "images": len(dataset),
            "gt_total": sum(len(d["gt"]) for d in dataset),
            "gt_per_class": {
                DISPLAY[c]: sum(1 for d in dataset for g in d["gt"] if g["cls"] == c)
                for c in CANONICAL
            },
        },
        "load_seconds": round(load_s, 3),
        "latency_cpu_ms": {
            "mean": round(1000 * float(np.mean(warm)), 1),
            "p50": round(1000 * float(np.percentile(warm, 50)), 1),
            "p95": round(1000 * float(np.percentile(warm, 95)), 1),
            "note": "CPU-only sandbox (no GPU); warm, per image incl. preprocessing in predict()",
        },
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "metrics": evaluate(preds_by_image, dataset),
        "operating_point": operating_point(preds_by_image, dataset, op_conf),
        "pr_sweep": pr_sweep(preds_by_image, dataset),
        "limitations": [
            "n=38 GT (Balloon 2, Bird 4): per-class figures for those classes are indicative only.",
            "Training data is not in the repo; train/test leakage vs training cannot be excluded from here.",
            "CPU-only timing; no RTX 3050 measurement was made in this environment.",
        ],
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--op-conf", type=float, default=0.25)
    ap.add_argument("--canonical-dedup", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run_benchmark(
        args.weights, args.imgsz, args.nms_iou, args.op_conf, args.canonical_dedup
    )
    suffix = "_dedup" if args.canonical_dedup else ""
    out = args.out or str(
        PROJECT_ROOT / "evaluation" / "results" / f"bench_{Path(args.weights).stem}_s{args.imgsz}_iou{args.nms_iou}{suffix}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    m, op = result["metrics"], result["operating_point"]
    print(f"weights={args.weights} imgsz={args.imgsz} nms_iou={args.nms_iou}")
    print(f"  mAP50={m['map50']:.3f}  mAP50-95={m['map50_95']:.3f}")
    for c in CANONICAL:
        pc = op["per_class"][DISPLAY[c]]
        print(f"  {DISPLAY[c]:9s} AP50={m['ap_per_class'][c]['0.50']:.3f} "
              f"P={pc['precision']:.3f} R={pc['recall']:.3f} (GT={pc['gt']})")
    print(f"  @conf{args.op_conf}: P={op['overall']['precision']:.3f} "
          f"R={op['overall']['recall']:.3f} F1={op['overall']['f1']:.3f} "
          f"FP={op['overall']['fp']} FN={op['overall']['fn']} dup={op['duplicate_pairs']} "
          f"ECE={op['calibration']['ece']}")
    print(f"  CPU latency mean={result['latency_cpu_ms']['mean']}ms load={result['load_seconds']}s -> {out}")


if __name__ == "__main__":
    main()
