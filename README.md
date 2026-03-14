# Pitch Visualizer

Pitch Visualizer turns a short paragraph into a storyboard by splitting text into scenes and generating an image for each scene with the Hugging Face Inference API.

## Stack

- Backend: Flask
- NLP: NLTK sentence tokenization
- Image generation: Hugging Face Inference API (Stable Diffusion XL)
- Frontend: Flask templates with HTML/CSS

## Project Structure

- app.py
- prompt_engine.py
- image_generator.py
- requirements.txt
- .env
- templates/index.html
- static/images/

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Add your key in .env:

   HF_API_KEY=your_huggingface_api_key

4. Run the app:

   python app.py

5. Open:

   http://127.0.0.1:5000

## Render Deployment

Build command:

pip install -r requirements.txt

Start command:

python app.py

Set environment variable in Render dashboard:

HF_API_KEY
