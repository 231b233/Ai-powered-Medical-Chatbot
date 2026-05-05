from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import tensorflow as tf
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load Models
try:
    cnn_model = tf.keras.models.load_model('medical_image_model.keras')
    logger.info("Successfully loaded CNN model.")
except Exception as e:
    logger.error(f"Failed to load CNN model: {e}")
    cnn_model = None

try:
    text_model_data = joblib.load('text_disease_model.pkl')
    text_model = text_model_data['model']
    text_columns = text_model_data['columns']
    logger.info("Successfully loaded Random Forest Text model.")
except Exception as e:
    logger.error(f"Failed to load Text model: {e}")
    text_model = None
    text_columns = None

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def query_groq_llama(system_prompt, messages_content, is_vision=False):
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct" if is_vision else "llama-3.3-70b-versatile"

    if is_vision:
        # Groq vision models do not support system prompts, so append it to the user message
        messages_content[0]["text"] = system_prompt + "\n\n" + messages_content[0]["text"]
        messages = [{"role": "user", "content": messages_content}]
    else:
        # Standard language models require content to be a simple string, not a list
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": messages_content[0]["text"]}
        ]

    response = requests.post(
        GROQ_API_URL,
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": 1000
        },
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=30
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        logger.error(f"Groq API Error: {response.status_code} - {response.text}")
        return f"Error from Groq API: {response.status_code}"

@app.post("/analyze_image")
async def analyze_image(image: UploadFile = File(...), query: str = Form(...)):
    try:
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        cnn_diagnosis = "CNN Model not available"
        
        if cnn_model:
            img_for_cnn = Image.open(io.BytesIO(image_content)).convert('RGB')
            img_for_cnn = img_for_cnn.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_for_cnn)
            img_array = tf.expand_dims(img_array, 0)
            predictions = cnn_model.predict(img_array)
            class_names = ['Healthy', 'Sick']
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = 100 * np.max(predictions[0])
            cnn_diagnosis = f"{predicted_class} (Confidence: {confidence:.2f}%)"

        augmented_query = f"{query}\n\n[Internal ML Diagnosis from CNN: {cnn_diagnosis}]"
        
        system_prompt = "You are an expert AI Medical Doctor. Look at the uploaded image and the Internal ML Diagnosis provided by our custom CNN. Explain the diagnosis and answer the user's question. IMPORTANT: End your response with a disclaimer stating you are an AI and this does not replace professional medical advice."
        messages_content = [
            {"type": "text", "text": augmented_query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]
        
        llama_answer = query_groq_llama(system_prompt, messages_content, is_vision=True)
        
        return JSONResponse(status_code=200, content={"ml_prediction": cnn_diagnosis, "llama_response": llama_answer})
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_text")
async def analyze_text(symptoms: str = Form(...), query: str = Form(...)):
    try:
        ml_diagnosis = "Text ML Model not available"
        
        if text_model and text_columns is not None:
            import pandas as pd
            # Create a single row dataframe with all zeros
            input_data = pd.DataFrame(0, index=[0], columns=text_columns)
            
            # Very basic NLP parsing: if symptom text is found in the column name, flag it
            symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
            for symptom in symptoms_list:
                for col in text_columns:
                    if symptom in col.lower():
                        input_data.at[0, col] = 1
            
            prediction = text_model.predict(input_data)
            ml_diagnosis = f"Predicted Disease: {prediction[0]}"
            
        augmented_query = f"Patient Symptoms: {symptoms}\nQuestion: {query}\n\n[Internal ML Diagnosis from Random Forest: {ml_diagnosis}]"
        
        system_prompt = "You are an expert AI Medical Doctor. Analyze the patient's text symptoms and the Internal ML Diagnosis provided by our custom Random Forest model. Give health guidance. IMPORTANT: End with a disclaimer stating you are an AI and this does not replace professional medical advice."
        messages_content = [
            {"type": "text", "text": augmented_query}
        ]
        
        llama_answer = query_groq_llama(system_prompt, messages_content, is_vision=False)
        
        return JSONResponse(status_code=200, content={"ml_prediction": ml_diagnosis, "llama_response": llama_answer})
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080)