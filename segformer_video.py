import cv2, torch, time
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import numpy as np

# ---- Model yükle ----
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).cuda().eval()

# ---- Video giriş/çıkış ----
video_path = "/home/ula-sak-n/Downloads/drone_20250826_140723.mp4"
cap = cv2.VideoCapture(video_path)

out_path = "/home/ula-sak-n/Personal/SkySeg/examples/output2_sky.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps_in = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

# ---- Sadece 20 saniye işle ----
max_frames = int(fps_in * 20)

frame_count = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count >= max_frames:
        break
    
    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    # Sky mask
    sky_mask = ((pred == 2) | (pred == 3)).astype("uint8") * 255
    sky_mask = cv2.resize(sky_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Overlay
    overlay = frame.copy()
    overlay[sky_mask == 255] = (0, 0, 255)
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
