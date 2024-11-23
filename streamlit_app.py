import requests
import json
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1/"
model_api = 'disaster_classifier'
headers = {
  'Content-Type': 'application/json'
}

payload = json.dumps({
  "text": [
    "there is a tsunami coming",
    "I am tired"
  ],
  "user_id": "email@email.com"
})


#response = requests.request("POST", url, headers=headers, data=payload)

st.title('Tweet classification')

text = st.text_area('Enter your text:')
user_id = st.text_input('Enter user id:', 'gareth@gareth.com')

data =  {
  'text': [text],
  "user_id": user_id
}

if st.button("Predict"):
  with st.spinner("Predicting..."):
    response = requests.post(API_URL+model_api, headers=headers, json=data)
    output = response.json()
  st.write(output)