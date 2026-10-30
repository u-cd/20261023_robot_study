"""固定Webカメラから、スポンジと置き場所のYOLO形式データを保存する。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid

import cv2

CLASSES = {0: "sponge", 1: "target"}


def yolo_box(box, width, height, class_id=0):
    if class_id not in CLASSES:
        raise ValueError("未知のクラスです。")
    x, y, w, h = box
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("画像内の対象物全体を囲んでください。")
    return f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}\n"


def save_example(frame, annotations, root, split, session):
    height, width = frame.shape[:2]
    label = "".join(yolo_box(box, width, height, class_id)
                    for class_id, box in sorted(annotations.items()))
    name = f"{session}_{uuid.uuid4().hex[:10]}"
    image = root / "images" / split / (name + ".jpg")
    annotation = root / "labels" / split / (name + ".txt")
    image.parent.mkdir(parents=True, exist_ok=True)
    annotation.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image), frame):
        raise OSError(f"画像を保存できません: {image}")
    annotation.write_text(label)
    with (root / "samples.jsonl").open("a") as out:
        out.write(json.dumps({"image": str(image.relative_to(root)), "split": split,
                              "session": session,
                              "objects": [{"class_id": c, "name": CLASSES[c], "box_xywh": box}
                                          for c, box in sorted(annotations.items())],
                              "captured_at": datetime.now().astimezone().isoformat()}) + "\n")
    return image


def annotate_frame(frame):
    """1フレーム上の両方の対象を囲み、まとめて保存する。Escで破棄。"""
    annotations = {}
    window = "Label: 1=sponge 2=target Enter=save Esc=cancel"
    try:
        while True:
            preview = frame.copy()
            for class_id, box in annotations.items():
                x, y, w, h = box
                color = (0, 200, 0) if class_id == 0 else (0, 150, 255)
                cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                cv2.putText(preview, CLASSES[class_id], (x, max(18, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(preview, "1: sponge | 2: target | Enter: save | Esc: cancel",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 0), 1)
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in [ord("1"), ord("2")]:
                class_id = 0 if key == ord("1") else 1
                roi_window = f"Select {CLASSES[class_id]} - Enter: accept / C: cancel"
                box = cv2.selectROI(roi_window, frame, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow(roi_window)
                if box[2] > 0 and box[3] > 0:
                    annotations[class_id] = list(map(int, box))
            elif key in [10, 13]:
                if annotations:
                    return annotations
                print("枠がありません。映っている対象を1・2で囲んでください。")
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description="Sで撮影・枠付け、Nで背景のみ保存、Qで終了")
    parser.add_argument("--camera", default="0", help="カメラ番号、または/dev/video0など")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", type=Path, default=Path("data/sponge"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "dataset.yaml"
    if (config_path.exists() and "1: target" not in config_path.read_text()
            and any((root / "images").glob("*/*.jpg"))):
        raise SystemExit("旧1クラスの画像があります。--output data/sponge_target を指定して新規収集するか、既存画像へ置き場所の枠を追加してください。")
    config = f"path: {json.dumps(str(root))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: sponge\n  1: target\n"
    (root / "dataset.yaml").write_text(config)
    for split in ["train", "val", "test"]:
        for kind in ["images", "labels"]:
            (root / kind / split).mkdir(parents=True, exist_ok=True)

    source = int(args.camera) if args.camera.isdigit() else args.camera
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        camera.release()
        raise SystemExit("カメラを開けません。接続・番号・他アプリでの使用を確認してください。")
    session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    saved = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("カメラから画像を取得できません。")
            preview = frame.copy()
            cv2.putText(preview, f"{args.split} | saved {saved} | S: label / N: empty / Q: quit",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 0), 1)
            cv2.imshow("Sponge dataset", preview)
            key = cv2.waitKey(20) & 0xFF
            if key in [ord("q"), 27] or cv2.getWindowProperty("Sponge dataset", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                annotations = annotate_frame(frame)
                if annotations is not None:
                    print(save_example(frame, annotations, root, args.split, session))
                    saved += 1
            elif key == ord("n"):
                print("背景のみ（スポンジも黒い置き場所も映っていないことを確認）:",
                      save_example(frame, {}, root, args.split, session))
                saved += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"保存: {saved}枚 / {args.split} / {root}")


if __name__ == "__main__":
    main()
