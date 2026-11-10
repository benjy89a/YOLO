"""
원본 대조용 확대 그림 만들기 → outputs/yolo/<사진이름>/checks/

yolo_service.py 로 결과를 만든 뒤 실행합니다. 결과 그림(이미 얼굴·번호판 모자이크됨)을 잘라서 쓰므로
여기서 만든 그림에도 개인정보가 드러나지 않습니다.
    .venv/Scripts/python.exe make_checks.py
"""
from pathlib import Path

import cv2
import numpy as np

import yolo_service as ys

OUT = ys.OUT_ROOT


def crop(path, xyxy, scale=1.0):
    x1, y1, x2, y2 = xyxy
    c = cv2.imread(str(path))[y1:y2, x1:x2]
    return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def side_by_side(*imgs, titles=()):
    h = max(i.shape[0] for i in imgs)
    out = []
    for i, img in enumerate(imgs):
        pad = np.full((h, img.shape[1], 3), 255, np.uint8)
        pad[: img.shape[0]] = img
        if i < len(titles):
            cv2.rectangle(pad, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(pad, titles[i], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.append(pad)
        out.append(np.full((h, 8, 3), 255, np.uint8))
    return np.hstack(out[:-1])


def save(name, folder, img):
    p = OUT / folder / "checks" / name
    cv2.imwrite(str(p), img)
    print("saved", p.relative_to(ys.ROOT))


# ── bus: conf 0.25 아래에 숨어 있던 오탐 후보 (주차금지 표지판 → stop sign 0.13)
_, low = ys.detect("input/bus.jpg", conf=0.05)
img = cv2.imread(str(OUT / "bus" / "original.jpg"))
for d in low["detections"]:
    if d["confidence"] < 0.25:
        x1, y1, x2, y2 = map(int, d["box_xyxy_px"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{d['class_name']} {d['confidence']:.2f}", (x1 + 2, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
save("low_conf_candidates_conf0.05.jpg", "bus", img)
save("stop_sign_false_positive.jpg", "bus", cv2.resize(img[200:400, 0:220], None, fx=2.5, fy=2.5))

# ── bus: 3번 사람(선글라스) 휴대폰 쥔 손 아래 마스크 구멍 (x 271~282, y 683~697) — 왼쪽 원본 / 오른쪽 마스크
o = cv2.imread(str(OUT / "bus" / "original.jpg"))[640:740, 230:320]
s = cv2.imread(str(OUT / "bus" / "segment" / "conf0.25.jpg"))[640:740, 230:320]
save("phone_missing_from_mask.jpg", "bus", cv2.resize(np.hstack([o, s]), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))

# ── 자동차: 멀리 있는 흰 차 — conf 0.50 에서 사라짐(미탐) + 마스크가 오른쪽 배경까지 퍼짐
car = "20161029_194728"
r = [1550, 800, 2250, 1350]
save("far_car_conf025_vs_conf050_vs_mask.jpg", car, side_by_side(
    crop(OUT / car / "detect" / "conf0.25.jpg", r), crop(OUT / car / "detect" / "conf0.50.jpg", r),
    crop(OUT / car / "segment" / "conf0.25.jpg", r),
    titles=("detect 0.25", "detect 0.50", "segment 0.25")))

# ── 야외 요리: 수저통 → cell phone 0.38 (오탐)
food = "20180309_190421"
save("tray_as_cell_phone.jpg", food, crop(OUT / food / "detect" / "conf0.25.jpg", [2100, 2300, 3300, 2988], 0.6))

# ── 야외 요리: 왼쪽 사람들 — 0.25 에서는 찾고, 0.50 에서는 놓침(미탐)
r = [0, 700, 1800, 2988]
save("left_people_conf025_vs_conf050.jpg", food, side_by_side(
    crop(OUT / food / "detect" / "conf0.25.jpg", r, 0.35), crop(OUT / food / "detect" / "conf0.50.jpg", r, 0.35),
    titles=("detect 0.25", "detect 0.50")))

# ── 야외 요리: 웍 위 면 → pizza 0.38, 웍 → bowl 0.31 (80개 클래스에 '면 요리·웍·냄비'가 없음)
save("wok_as_pizza_and_bowl.jpg", food, crop(OUT / food / "segment" / "conf0.25.jpg", [1300, 1100, 3700, 2400], 0.4))

# ── 야외 요리: 사람 두 명 사이의 좁은 박스 person 0.37 (중복/애매 → 확인 필요)
save("narrow_person_box_between_two.jpg", food, crop(OUT / food / "detect" / "conf0.25.jpg", [3300, 0, 4300, 1100], 0.5))
