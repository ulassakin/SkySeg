# SkySeg – Sky Segmentation for Drone Detection

**SkySeg** is a lightweight tool that extracts the **sky region** from aerial videos.  
It is designed as a preprocessing step to make **drone detection at high altitudes** easier by reducing background noise. It uses **SegFormer-B0 pretrained on ADE20K** to generate binary sky masks. This project was developed as part of a drone detection pipeline, where isolating the sky helps reduce background noise for small-object detection.

## 🚀 Features
- Segment the **sky** region in any image or video.
- Overlay masks with transparent color for visualization.
- Benchmark FPS to evaluate real-time performance.
- Runs on both CPU and GPU (optimized for CUDA).

---

## ⚙️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/SkySeg.git
cd SkySeg
pip install -r requirements.txt
```


## Usage 
- Run sky segmentation on a video:

```bash
python segformer_video.py
```

Modify inside the script:

-video_path → your input video path.

-out_path → output video name.

At the end, you’ll see:

-output_sky.mp4 → overlayed segmentation video.

-FPS results printed in the console.

## 📌 Notes:
- Model is pretrained on ADE20K (150 classes).
- The sky class ID is usually 2 or 3, depending on label reduction during training.
- If masks look blocky, apply Gaussian blur for smoother edges although this will reduce the fps.
