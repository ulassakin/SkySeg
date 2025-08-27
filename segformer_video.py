import cv2, torch, time
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from pathlib import Path
import numpy as np

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).cuda().eval()


video_path = "/home/ula-sak-n/Downloads/test1.mp4"   # kendi videonu buraya koy
cap = cv2.VideoCapture(video_path)

out_path = "/home/ula-sak-n/Personal/SkySeg/examples/output1_sky.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps_in = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))


frame_count = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    
    sky_mask = ((pred == 2) | (pred == 3)).astype("uint8") * 255
    

    sky_mask = cv2.resize(sky_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    
    overlay = frame.copy()
    overlay[sky_mask == 255] = (0, 0, 255)  # gökyüzünü kırmızıya boya
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    writer.write(blended)
    frame_count += 1
    
t1 = time.time()
cap.release()
writer.release()

avg_time = (t1 - t0) / frame_count
print(f"Processed {frame_count} frames in {(t1 - t0):.2f} sec")
print(f"Avg per frame: {avg_time:.3f} sec -> {1/avg_time:.1f} FPS")
print(f"Saved video: {out_path}")
