from io import BytesIO

import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from bestmlops.model import classify_digit

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # allow all origins (in real life you should specify the frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/infer/")
async def infer(file: UploadFile):
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    predictions = classify_digit(image)
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8071)
