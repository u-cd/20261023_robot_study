"""Background YOLO inference; overlay and boxes always belong to the same frame."""
import threading
import time

import cv2


class LiveDetector:
    def __init__(self, camera, weights):
        self.camera = camera
        self.weights = weights
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.result = None
        self.error = '検出モデルを読み込み中です。'
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(2)
            model = YOLO(self.weights)
        except Exception as error:
            with self.lock:self.error = '検出モデルの読み込みに失敗しました: ' + str(error)
            return
        while not self.stop.is_set():
            try:
                frame, stamp, sequence = self.camera.snapshot(fresh=True)
                captured = time.monotonic()
                prediction = model.predict(frame, imgsz=320, conf=.25, device='cpu', verbose=False)[0]
                detections = []
                for box in prediction.boxes:
                    name = model.names[int(box.cls.item())]
                    if name not in ('sponge', 'target'):continue
                    confidence = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()
                    detections.append({'class':name, 'confidence':confidence, 'box_xyxy_px':xyxy})
                    x1,y1,x2,y2 = map(round,xyxy)
                    color = (70,210,70) if name=='sponge' else (220,100,220)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    label = f'{"sponge" if name=="sponge" else "coin"} {confidence:.2f}'
                    cv2.putText(frame,label,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
                with self.lock:
                    self.result = (frame, stamp, sequence, captured, detections)
                    self.error = None
            except Exception as error:
                with self.lock:
                    self.result = None
                    self.error = 'ライブ検出を更新できません: ' + str(error)
            self.stop.wait(.1)

    def snapshot(self):
        with self.lock:
            if self.error or self.result is None:
                raise RuntimeError(self.error or '検出準備中です。')
            frame, stamp, seq, captured, detections = self.result
            if time.monotonic()-captured > 2:
                raise RuntimeError('ライブ検出の画像が古いため、更新を待っています。')
            return frame.copy(), stamp, seq

    def observation(self):
        with self.lock:
            if self.error or self.result is None:
                raise RuntimeError(self.error or '検出準備中です。')
            frame, stamp, sequence, captured, boxes = self.result
            if time.monotonic()-captured>1:
                raise RuntimeError('検出画像が古いため予測を更新できません。')
            height,width=frame.shape[:2]
            detections={}
            for box in boxes:
                name=box['class']
                if box['confidence']<=detections.get(name,{}).get('confidence',-1):continue
                x1,y1,x2,y2=box['box_xyxy_px']
                detections[name]={'center_normalized':[(x1+x2)/2/width,(y1+y2)/2/height],
                    'confidence':box['confidence'],'box_xyxy_px':list(box['box_xyxy_px'])}
            return {'detections':detections,'sequence':sequence,'captured_at':stamp}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)
