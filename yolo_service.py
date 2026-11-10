"""
YOLO 서비스 핵심 코드 (과제 A)

사진 1장을 넣으면
  1) 탐지(detect)  : yolo26n.pt      → 박스 · 클래스 · 점수
  2) 분할(segment) : yolo26n-seg.pt  → 마스크(윤곽을 따라 칠한 영역)
를 conf(점수 임계값)별로 실행하고, 결과를 outputs/yolo/<사진이름>/ 에 저장합니다.

명령줄 사용 예:
    .venv/Scripts/python.exe yolo_service.py input/bus.jpg
    .venv/Scripts/python.exe yolo_service.py input/my_photo.jpg --conf 0.25 0.50
"""

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_ROOT = ROOT / "outputs" / "yolo"

# 과제에서 정한 고정 설정
IMGSZ = 640       # 모델에 넣기 전에 사진을 이 크기(긴 변 640px)에 맞춰 줄여서 봅니다
DEVICE = "cpu"    # 그래픽카드 대신 CPU로 계산합니다

PRIVACY_FILE = ROOT / "privacy_regions.json"

# 입력 사진 출처와 공유 조건 (manifest.json 에 기록)
INPUT_SOURCES = {
    "bus.jpg": "Ultralytics 예제 이미지 (https://ultralytics.com/images/bus.jpg), 공개 예제",
    "20161029_194728.jpg": "직접 촬영(본인). 번호판 모자이크 후 과제 제출용으로만 공유. 원본은 제출하지 않음",
    "20180309_190421.jpg": "직접 촬영(본인). 촬영된 분들의 공유 동의는 받지 않아 얼굴 모자이크 후 과제 제출용으로만 공유. 원본은 제출하지 않음",
}

_models = {}


# ─────────────────────────────────────────────
# 0) 개인정보 가리기 (얼굴 · 번호판)
#    모델에는 원본을 그대로 넣고(결과가 바뀌지 않게), 저장하는 '그림'에만 모자이크합니다.
# ─────────────────────────────────────────────
def privacy_regions(image_path):
    if not PRIVACY_FILE.exists():
        return []
    return json.loads(PRIVACY_FILE.read_text(encoding="utf-8")).get(Path(image_path).name, [])


def mosaic(img, regions):
    """영역을 아주 작게 줄였다가 다시 키우면 큰 네모 조각(모자이크)만 남아서 알아볼 수 없게 됩니다.
    조각 크기는 영역 짧은 변의 1/8 (최소 24px) → 큰 얼굴일수록 조각도 커서 윤곽이 남지 않아요."""
    out = img.copy()
    h, w = out.shape[:2]
    for reg in regions:
        x1, y1, x2, y2 = reg["xyxy"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        block = max(24, min(x2 - x1, y2 - y1) // 8)
        small = cv2.resize(roi, (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def base_image(image_path):
    """결과 그림의 '바탕'이 될 원본. 얼굴·번호판을 먼저 모자이크해 두고, 그 위에 박스·라벨을 그립니다
    (그래야 라벨 글씨는 가려지지 않아요)."""
    return mosaic(cv2.imread(str(image_path)), privacy_regions(image_path))


def load_model(name):
    """모델은 한 번만 불러와서 재사용합니다 (매번 불러오면 느려요)."""
    if name not in _models:
        _models[name] = YOLO(str(MODEL_DIR / name))
    return _models[name]


# ─────────────────────────────────────────────
# 1) 탐지: 박스 · 클래스 · 점수
# ─────────────────────────────────────────────
def detect(image_path, conf):
    model = load_model("yolo26n.pt")
    r = model.predict(str(image_path), imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]

    h, w = r.orig_shape
    detections = []
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 원본 사진 기준 픽셀 좌표 (왼쪽 위, 오른쪽 아래)
        cls_id = int(box.cls[0])
        detections.append({
            "id": i,
            "class_id": cls_id,
            "class_name": r.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "box_xyxy_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "box_wh_px": [round(x2 - x1, 1), round(y2 - y1, 1)],
        })

    data = {
        "task": "detect",
        "model": "yolo26n.pt",
        "image": str(Path(image_path).as_posix()),
        "image_size_hw": [h, w],
        "coord_format": "xyxy, 원본 이미지 픽셀 단위 (0~1 정규화 아님)",
        "settings": {"imgsz": IMGSZ, "conf": conf, "device": DEVICE},
        "count_by_class": _count(detections),
        "detections": detections,
    }
    return r, data


# ─────────────────────────────────────────────
# 2) 분할: 마스크
# ─────────────────────────────────────────────
def segment(image_path, conf):
    model = load_model("yolo26n-seg.pt")
    # retina_masks=True: 마스크를 원본 사진 크기 그대로 받습니다 (기본값은 모델 입력 크기라 다시 늘려야 함)
    r = model.predict(str(image_path), imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False, retina_masks=True)[0]

    h, w = r.orig_shape
    objects, masks = [], []
    pixel_masks = r.masks.data.cpu().numpy() if r.masks is not None else []   # (객체 수, 높이, 너비), 값 0~1
    for i, (box, pm) in enumerate(zip(r.boxes, pixel_masks)):
        # 모델이 '대상일 확률이 0.5보다 크다'고 본 픽셀 = 255(흰색), 나머지 = 0(검정)
        mask = (pm > 0.5).astype(np.uint8) * 255
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        mask_area = int((mask > 0).sum())
        cls_id = int(box.cls[0])
        objects.append({
            "id": i,
            "class_name": r.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "box_xyxy_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "box_area_px": int(box_area),
            "mask_area_px": mask_area,
            # 박스 안에서 마스크가 차지하는 비율 → 1보다 작을수록 박스에 배경이 많이 섞였다는 뜻
            "mask_over_box": round(mask_area / box_area, 3) if box_area else None,
            "mask_file": None,  # 저장할 때 채움
        })

    data = {
        "task": "segment",
        "model": "yolo26n-seg.pt",
        "image": str(Path(image_path).as_posix()),
        "image_size_hw": [h, w],
        "settings": {"imgsz": IMGSZ, "conf": conf, "device": DEVICE},
        "count_by_class": _count(objects),
        "objects": objects,
    }
    return r, data, masks


def mask_overlay(image_path, masks, objects, alpha=0.45):
    """원본 위에 마스크를 반투명 색으로 겹쳐 그립니다."""
    img = base_image(image_path)
    over = img.copy()
    # 사진이 클수록 글씨와 선도 크게 (bus 810px 기준 → 휴대폰 사진 5312px도 잘 보이게)
    k = max(img.shape[:2]) / 1000
    font, thick = 0.7 * k, max(2, int(2 * k))
    rng = np.random.default_rng(0)
    for m, o in zip(masks, objects):
        color = rng.integers(60, 255, size=3).tolist()
        over[m > 0] = color
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, thick)
    out = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for m, o in zip(masks, objects):
        ys, xs = np.nonzero(m)
        if len(xs):
            text = f"{o['id']}:{o['class_name']} {o['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
            # 라벨은 마스크의 '맨 위 점' 근처에 붙입니다 (다른 물체 라벨과 헷갈리지 않게)
            top = int(ys.min())
            x = int(np.median(xs[ys == top])) - tw // 2
            x, y = min(max(0, x), out.shape[1] - tw), max(top - 6, th + 4)
            cv2.rectangle(out, (x, y - th - 4), (x + tw, y + 4), (0, 0, 0), -1)   # 글씨 뒤 검은 띠
            cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)
    return out


# ─────────────────────────────────────────────
# 3) conf 두 개 비교: 무엇이 사라졌나?
# ─────────────────────────────────────────────
def compare(det_low, det_high):
    lo, hi = det_low["settings"]["conf"], det_high["settings"]["conf"]
    # 높은 conf 결과는 낮은 conf 결과의 '부분집합'입니다 (같은 모델·같은 사진이면
    # 점수가 hi 이상인 박스만 남기 때문). 그래서 lo 이상 hi 미만 점수를 가진 박스가 '사라진 것'입니다.
    removed = [d for d in det_low["detections"] if d["confidence"] < hi]
    return {
        "conf_low": lo,
        "conf_high": hi,
        "count_low": len(det_low["detections"]),
        "count_high": len(det_high["detections"]),
        "removed_when_raising_conf": removed,
    }


# ─────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────
def run(image_path, confs=(0.25, 0.50)):
    """
    결과 폴더 구조 (사진 1장마다):
        outputs/yolo/<사진이름>/
            original.jpg              원본 사본 (EXIF·GPS 정보는 저장되지 않음)
            detect/conf0.25.jpg|json  탐지 결과 그림 + 박스 데이터
            segment/conf0.25.jpg|json 분할 결과 그림 + 마스크 정보
            segment/masks/conf0.25/00_person.png ...  객체별 마스크
            compare_conf.json         conf를 올렸을 때 사라진 박스
            manifest.json             실행 기록
            checks/                   노트북에서 확인용으로 만든 그림 (확대 등)
    """
    image_path = Path(image_path)
    out_dir = OUT_ROOT / image_path.stem
    for sub in ("detect", "segment/masks", "checks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    # 과제 요구: outputs/yolo/ 에 실행 코드도 함께 저장 (결과와 그 결과를 만든 코드를 한곳에)
    for src in (Path(__file__), PRIVACY_FILE):
        if src.exists():
            shutil.copy2(src, OUT_ROOT / src.name)
    cv2.imwrite(str(out_dir / "original.jpg"), base_image(image_path))

    det_results, files = {}, ["original.jpg"]
    for conf in confs:
        tag = f"conf{conf:.2f}"

        r, det = detect(image_path, conf)
        cv2.imwrite(str(out_dir / "detect" / f"{tag}.jpg"), r.plot(img=base_image(image_path)))
        _dump(det, out_dir / "detect" / f"{tag}.json")
        det_results[conf] = det
        files += [f"detect/{tag}.jpg", f"detect/{tag}.json"]

        r, seg, masks = segment(image_path, conf)
        (out_dir / "segment" / "masks" / tag).mkdir(parents=True, exist_ok=True)
        for o, m in zip(seg["objects"], masks):
            name = f"segment/masks/{tag}/{o['id']:02d}_{o['class_name'].replace(' ', '_')}.png"
            cv2.imwrite(str(out_dir / name), m)
            o["mask_file"] = name
        cv2.imwrite(str(out_dir / "segment" / f"{tag}.jpg"), mask_overlay(image_path, masks, seg["objects"]))
        _dump(seg, out_dir / "segment" / f"{tag}.json")
        files += [f"segment/{tag}.jpg", f"segment/{tag}.json", f"segment/masks/{tag}/ ({len(masks)}개)"]

    cmp = compare(det_results[min(confs)], det_results[max(confs)])
    _dump(cmp, out_dir / "compare_conf.json")
    files.append("compare_conf.json")

    write_manifest(image_path, out_dir, confs, files)
    return out_dir, det_results, cmp


def write_summary():
    """outputs/yolo/summary.json: 사진별 결과를 한눈에 보는 목록."""
    rows = []
    for d in sorted(p for p in OUT_ROOT.iterdir() if (p / "compare_conf.json").exists()):
        cmp = json.loads((d / "compare_conf.json").read_text(encoding="utf-8"))
        row = {"image": d.name, "folder": str(d.relative_to(ROOT).as_posix())}
        for conf in (cmp["conf_low"], cmp["conf_high"]):
            det = json.loads((d / "detect" / f"conf{conf:.2f}.json").read_text(encoding="utf-8"))
            seg = json.loads((d / "segment" / f"conf{conf:.2f}.json").read_text(encoding="utf-8"))
            row[f"conf{conf:.2f}"] = {"detect": det["count_by_class"], "segment": seg["count_by_class"]}
        row["removed_when_raising_conf"] = [(x["class_name"], x["confidence"]) for x in cmp["removed_when_raising_conf"]]
        rows.append(row)
    _dump(rows, OUT_ROOT / "summary.json")
    return rows


def write_manifest(image_path, out_dir, confs, files):
    """나중에 다른 사람이 똑같이 다시 돌릴 수 있도록 '무엇을, 어떻게' 돌렸는지 기록합니다."""
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(Path(image_path).as_posix()),
        "input_source": INPUT_SOURCES.get(Path(image_path).name, "직접 기입: 출처 / 공유 동의 여부"),
        "input_size_hw": list(cv2.imread(str(image_path)).shape[:2]),
        "privacy": {
            "exif_gps": "출력 이미지는 OpenCV로 다시 저장해 EXIF(GPS 포함)가 들어가지 않음",
            "mosaic_regions": privacy_regions(image_path),
            "mosaic_applies_to": "저장된 결과 그림(original/detect/segment jpg)에만 적용. 모델 입력과 JSON 좌표·마스크는 원본 기준",
            "original_input_submitted": False if privacy_regions(image_path) else None,
        },
        "models": {
            "detect": str((MODEL_DIR / "yolo26n.pt").relative_to(ROOT).as_posix()),
            "segment": str((MODEL_DIR / "yolo26n-seg.pt").relative_to(ROOT).as_posix()),
        },
        "packages": {
            "python": sys.version.split()[0],
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "run_location": {
            "where": "local",
            "os": f"{platform.system()} {platform.release()}",
            "venv": str(Path(sys.executable).as_posix()),
        },
        "settings": {"imgsz": IMGSZ, "device": DEVICE, "conf": list(confs)},
        "code": "yolo_service.py",
        "outputs": files,
    }
    _dump(manifest, out_dir / "manifest.json")


def _count(items):
    c = {}
    for it in items:
        c[it["class_name"]] = c.get(it["class_name"], 0) + 1
    return c


def _dump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="사진 파일 (여러 장 가능)")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.50])
    a = ap.parse_args()
    for image in a.images:
        out_dir, dets, cmp = run(image, tuple(a.conf))
        print("=" * 60, "\n저장 위치:", out_dir)
        for conf, d in dets.items():
            print(f"\n[detect conf={conf}] 개수 {len(d['detections'])}  {d['count_by_class']}")
            for x in d["detections"]:
                print(f"  #{x['id']} {x['class_name']:<12} {x['confidence']:.3f}  box={x['box_xyxy_px']}")
        print(f"\n[conf {cmp['conf_low']} → {cmp['conf_high']}] 사라진 박스:")
        for x in cmp["removed_when_raising_conf"] or [{"class_name": "(없음)", "confidence": 0}]:
            print(f"  {x['class_name']} {x['confidence']:.3f}")
    write_summary()
    print("\n사진별 요약:", OUT_ROOT / "summary.json")
