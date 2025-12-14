# SMS Spam Detection using NLP & Deep Learning

A web application that classifies SMS messages as spam or legitimate (ham).


## 🎯 Project Overview

This project implements a spam detection system with:
- **Deep Learning Model**: Bidirectional LSTM with embedding layer
- **NLP Preprocessing**: Text cleaning, tokenization, stopword removal, lemmatization
- **Web Interface**: Flask application with modern, responsive UI

## ✨ Features

- 📊 **High Accuracy**: 88%+ accuracy on test data
- 🧠 **Deep Learning**: BiLSTM architecture with 100-dim embeddings
- 🎨 **Modern UI**: Gradient design with animations and visual feedback
- 📱 **Responsive**: Works seamlessly on desktop and mobile devices
- ⚡ **Real-time**: Instant predictions with confidence visualization

## 🏗️ Model Architecture

```
Input Layer (100 tokens)
    ↓
Embedding Layer (1000 vocab, 50 dimensions)
    ↓
Bidirectional LSTM (32 units, L2 regularization)
    ↓
Dropout (0.5)
    ↓
Dense Layer (1 unit, sigmoid activation)
    ↓
Output (Spam probability)
```

**Training Details:**
- Dataset: 5,572 SMS messages
- Class Distribution: 87% ham, 13% spam
- Optimizer: Adam
- Loss: Binary Crossentropy
- Early Stopping: 3 epochs patience
- Validation Split: 30%

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spam-detection-nlp.git
cd spam-detection-nlp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (if not already installed)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Running the Application

1. **Navigate to deployment folder**
```bash
cd deployment
```

2. **Start the Flask server**
```bash
python app.py
```

3. **Open your browser**
```
http://127.0.0.1:5000
```

## 💻 Usage

### Web Interface

1. Enter an SMS message in the text area (3-1000 characters)
2. Click "Analyze Message"
3. View the prediction with confidence score

### Example Messages

**Spam:**
```
URGENT! You have won a $1000 gift card. Call now to claim your prize!
```

**Ham:**
```
Hey, are you free for dinner tonight? Let me know what time works.
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98%+ |
| Precision (Spam) | 100% |
| Recall (Spam) | 84% |
| F1-Score (Spam) | 91% |

## 🛠️ Technologies Used

**Machine Learning & NLP:**
- TensorFlow 2.8.0
- Keras
- NLTK
- Scikit-learn

**Web Development:**
- Flask
- HTML5/CSS3
- JavaScript

**Data Processing:**
- Pandas
- NumPy

**Visualization:**
- Matplotlib
- Seaborn

## 📝 Preprocessing Pipeline

1. **Text Cleaning**: Lowercase, remove URLs, punctuation, digits
2. **Tokenization**: Split text into words
3. **Stopword Removal**: Filter common English words
4. **Lemmatization**: Reduce words to base form
5. **Vectorization**: Convert to sequences with Keras tokenizer
6. **Padding**: Standardize sequence length to 100 tokens

## 🔬 Development Process

1. **EDA**: Analyzed class distribution, word frequencies, n-grams
2. **Preprocessing**: Built NLP pipeline with NLTK
3. **Feature Engineering**: TF-IDF for baseline, embeddings for deep learning
4. **Model Training**: Tested Naive Bayes, Random Forest, SVM, BiLSTM
5. **Model Selection**: BiLSTM chosen for best performance
6. **Deployment**: Flask web app with modern UI

## 🚧 Troubleshooting

**Model Loading Error:**
```bash
# Downgrade protobuf
pip install protobuf==3.20.3
```

**NumPy Compatibility Warning:**
```bash
# Use NumPy 1.x
pip install "numpy<2.0" --force-reinstall
```

**Import Error (src module):**
- Ensure `__init__.py` exists in project root
- Run app from `deployment/` folder