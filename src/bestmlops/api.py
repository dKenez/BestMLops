from io import BytesIO
from typing import Union

import numpy as np
from fastapi import FastAPI, UploadFile
from PIL import Image

from bestmlops.model import classify_digit

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/infer/")
async def infer(file: UploadFile):
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    predictions = classify_digit(image)
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8071)
