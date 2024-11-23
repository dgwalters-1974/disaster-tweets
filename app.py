import warnings
warnings.filterwarnings('ignore')

from scripts.data_model import NLPDataInput, NLPDataOutput#, ImageDataInput, ImageDataOutput
from scripts import s3

from fastapi import FastAPI
from fastapi import Request
import uvicorn
import os
import time

import torch
from transformers import pipeline
from transformers import AutoImageProcessor #-> like Tokenizer

app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


### Load models from AWS ###

force_download = False # False

model_name = 'tinybert-disaster-tweet'
local_path = 'ml-models/'+model_name
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
tweeter_model = pipeline('text-classification', model=local_path, device=device)


### API Section ###

@app.get("/")
def read_root():
    return "Hello! I am up!!!"


@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = tweeter_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text = data.text,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)

    return output


if __name__=="__main__":
    uvicorn.run(app="app:app", port=8000, reload=True, host="127.0.0.1")