import argparse
import io
import logging
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.transforms import Compose

from api import PlantPredictor
from config import Config
from data import PlantDataset, create_val_transforms
from models import ResNet50

app = FastAPI()

transform: Compose
predictor: PlantPredictor
logger: logging.Logger


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info("New file recieved")
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Use existing predictor
        predictions = predictor.predict(input_tensor)[0]
        return JSONResponse(
            {
                "class_id": predictions["class_index"],
                "confidence": predictions["confidence"],
                "class_name": predictions["class_name"],
            }
        )
    except Exception:
        logger.exception("Error")
        raise HTTPException(status_code=400)


def main() -> None:
    global app, transform, predictor, logger
#  api --config configs/train_config.yaml
    parser = argparse.ArgumentParser(
        description="API to predict a plant class",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/train_config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config)

    model_path = Path("models/best_model/")
    model = ResNet50.from_pretrained(model_path)
    class_names = PlantDataset.LABELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = PlantPredictor(model, class_names, device)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    transform = create_val_transforms(config.transform_config)

    uvicorn.run(app, host="0.0.0.0", port=config.api_config.port)


if __name__ == "__main__":
    main()
