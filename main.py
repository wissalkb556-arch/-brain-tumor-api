from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io, base64, os
from PIL import Image

from model_utils import load_model, GradCAM, predict_and_explain
from report_generator import generate_report

app = FastAPI(title="Brain Tumor Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "best_model.pth")
model = load_model(WEIGHTS_PATH)
gradcam = GradCAM(model, model.layer4)


@app.get("/")
def health_check():
    return {"status": "API en ligne", "model": "ResNet18 fine-tuné"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = predict_and_explain(model, gradcam, image)
    report = generate_report(result)

    heatmap_img = Image.fromarray(result["heatmap_overlay"])
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse({
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "benign_malignant": result["benign_malignant"],
        "heatmap_image_base64": heatmap_b64,
        "report": report,
    })
