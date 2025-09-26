from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="Face Feature Extraction API")

# --- Load models ---
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet.classifier = nn.Identity()
mobilenet.eval()

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval()

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.post("/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    model_name: str = Query("mobilenet", enum=["mobilenet", "resnet"])
):
    try:
        # Load image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)

        # Extract embedding
        with torch.no_grad():
            if model_name == "mobilenet":
                embedding = mobilenet(input_tensor).numpy().tolist()[0]
            else:
                embedding = resnet(input_tensor).numpy().tolist()[0]

        return JSONResponse(content={"embedding": embedding, "model": model_name})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"status": "Face Feature Extraction API running"}
