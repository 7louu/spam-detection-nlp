from flask import Flask, render_template, request
import pickle
import os
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences
from src.preprocessing import preprocess_pipeline

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,'..', 'models', 'best_spam_detect_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, '..', 'models', 'tokenizer.pickle')

print("Loading model and tokenizer...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfuly!")

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded successfully!")

MAX_WORDS = 1000
MAX_LEN = 100

def predict_spam(text):
    try:
      processed_text = preprocess_pipeline(text)

      sequence = tokenizer.texts_to_sequences([processed_text])

      padded = pad_sequences(sequence, maxlen=MAX_LEN)

      prediction = model.predict(padded, verbose=0)

      probability = float(prediction[0][0])

      label = "Spam" if probability > 0.5 else "Ham"

      confidence = probability if label == "Spam" else 1 - probability

      return label, confidence * 100
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
      message = request.form['message']
      
      if not message or message.strip() == "":
        return render_template('index.html', error='Input cannot be empty. Please enter a valid message.')
      if len(message.strip()) < 3:
        return render_template('index.html', error='Input too short. Please enter a longer message.')
      if len(message) > 1000:
        return render_template('index.html', error='Input too long. Please enter a shorter message (max 1000 characters).')
      
      label, confidence = predict_spam(message)
      if label is None:
        return render_template('index.html', error='An error occurred during prediction. Please try again.')

      return render_template('index.html', message=message, prediction=label, confidence=round(confidence, 2)) 
    
    except Exception as e:
       return render_template('index.html', error=f'An unexpected error occurred: {str(e)}')
      

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)