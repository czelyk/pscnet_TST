from fastapi import FastAPI, HTTPException, UploadFile, File
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from run import Pscc  # Import Pscc class from run.py
from utils.config import get_pscc_args
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model when app starts
@app.on_event("startup")
def load_model():
    global pscc
    args = get_pscc_args()
    pscc = Pscc(args)

# Create a thread pool executor for running CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=3)

# Asynchronous wrapper to offload the prediction task to the executor
async def run_in_executor(image: torch.Tensor):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, pscc.predict, image)

# Define a FastAPI POST endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load the image from the uploaded file
        image = Image.open(BytesIO(await file.read()))
        image = image.convert("RGB")  # Ensure 3 channels
        image = np.array(image)
        image = torch.tensor(image).float().unsqueeze(0).permute(0, 3, 1, 2)  # Convert to CxHxW format

        # Normalize image values to [0, 1]
        image = image / 255.0

        # Get the prediction asynchronously (base64 result and confidence score)
        pred_result = await run_in_executor(image)

        # Ensure the model returns a tuple: (base64_string, confidence_score)
        if isinstance(pred_result, tuple) and len(pred_result) == 2:
            base64_image, confidence_score = pred_result
        else:
            raise ValueError("Model did not return a tuple (base64_image, confidence_score)")

        # Normalize confidence score (adjust divisor if needed)
        normalized_score = max(0.0, min(confidence_score / 100.0, 1.0))  # Adjust divisor based on your model

        # Return formatted response
        return {
            "Confidence Score": round(confidence_score * 0.1 , 10),
            "Base64 Image": base64_image
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
