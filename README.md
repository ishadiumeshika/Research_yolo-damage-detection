# Research_yolo-damage-detection

Final year research project for packaging and damage detection.

## Demo Video

This project includes a demonstration video showing the YOLOv11 model detecting package damage and shipping labels.

📹 Demo video: `360_video.avi` and `damaged_package_vide`

## Web App

The repository also includes a web application for reviewing a short 360 package video, asking for package details, and returning one of three verdicts:

- `package can be shipped`
- `normal`
- `package is not suitable for shipping`

The backend compares the declared package details against detections from the trained YOLO model, checks for visible damage, and stores each review in MongoDB when configured.

## What it checks

- Package weight versus the configured shipping limit
- Declared handling labels such as fragile, keep dry, shipping label, and this side up
- Visible package damage
- Whether the user-declared package cues appear in the uploaded video

The current model can only verify visible packaging cues. It cannot confirm hidden contents that are not visible in the video.

## Setup

1. Copy `.env.example` to `.env` and fill in the values.
2. The workspace already contains the trained weights at `best.pt`. Keep that file in the project root, or set `MODEL_WEIGHTS` to another `.pt` file if you move it.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
uvicorn app.main:app --reload
```

5. Open `http://127.0.0.1:8000`.

## MongoDB

Set `MONGODB_URI` to your MongoDB connection string in `.env`. The app stores each analysis in the `analyses` collection of the configured database.

## Notes

- Video uploads are limited to 120 seconds by default.
- The shipping weight threshold defaults to 30 kg and can be changed in `.env`.
- If the trained weights are missing, the UI still loads, but analysis will not produce detections until the model path is configured.
