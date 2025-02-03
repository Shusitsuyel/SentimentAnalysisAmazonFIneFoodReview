from flask import Flask, request, jsonify, render_template_string
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')



app = Flask(__name__)

# Sentiment label mapping
SENTIMENT_LABELS = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# Load model and vectorizer
model = joblib.load('review_score_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|[^\w\s]|\d', '', text)
    tokens = nltk.word_tokenize(text)  # Uses the standard `punkt` tokenizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Review Sentiment Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; text-align: center; }
                .container { max-width: 800px; margin: 0 auto; }
                .form-group { margin-bottom: 20px; }
                textarea { 
                    width: 100%; 
                    padding: 15px; 
                    border: 2px solid #3498db; 
                    border-radius: 8px; 
                    font-size: 16px;
                    resize: vertical;
                }
                button { 
                    background: #3498db; 
                    color: white; 
                    padding: 12px 25px; 
                    border: none; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    font-size: 16px;
                    transition: background 0.3s;
                }
                button:hover { background: #2980b9; }
                .result { 
                    margin-top: 30px; 
                    padding: 20px; 
                    border-radius: 8px; 
                    display: none;
                }
                .positive { background: #d4edda; color: #155724; }
                .neutral { background: #fff3cd; color: #856404; }
                .negative { background: #f8d7da; color: #721c24; }
                .spinner {
                    display: none;
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📝 Review Sentiment Analyzer</h1>
                
                <form id="reviewForm" onsubmit="handleSubmit(event)">
                    <div class="form-group">
                        <textarea 
                            name="review" 
                            rows="5" 
                            placeholder="Write your review here..."
                            required
                        ></textarea>
                    </div>
                    <button type="submit">Analyze Sentiment</button>
                </form>

                <div class="spinner" id="spinner"></div>
                
                <div id="result" class="result"></div>
            </div>

            <script>
                function handleSubmit(e) {
                    e.preventDefault();
                    const form = e.target;
                    const resultDiv = document.getElementById('result');
                    const spinner = document.getElementById('spinner');
                    
                    // Show spinner
                    spinner.style.display = 'block';
                    resultDiv.style.display = 'none';

                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            review: form.review.value
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Hide spinner
                        spinner.style.display = 'none';
                        
                        // Display results
                        if (data.error) {
                            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                        } else {
                            resultDiv.innerHTML = `
                                <h3>Analysis Results:</h3>
                                <p>Score: ${data.score}/4</p>
                                <p>Sentiment: <strong>${data.sentiment}</strong></p>
                            `;
                            resultDiv.className = `result ${data.sentiment.toLowerCase()}`;
                        }
                        resultDiv.style.display = 'block';
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        spinner.style.display = 'none';
                        resultDiv.innerHTML = 'Error processing your request. Please try again.';
                        resultDiv.style.display = 'block';
                    });
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review_text = data.get('review', '')
        
        if not review_text:
            return jsonify({'error': 'No review provided'}), 400
        
        print(f"Review Text: {review_text}")  # Debugging
        
        # Clean and predict
        cleaned_review = clean_text(review_text)
        print(f"Cleaned Review: {cleaned_review}")  # Debugging
        
        review_tfidf = vectorizer.transform([cleaned_review])
        print(f"TF-IDF Vector: {review_tfidf}")  # Debugging
        
        predicted_score = model.predict(review_tfidf)[0]
        print(f"Predicted Score: {predicted_score}")  # Debugging
        
        # Get sentiment label
        sentiment = SENTIMENT_LABELS.get(predicted_score, "Unknown")
        print(f"Sentiment: {sentiment}")  # Debugging
        
        return jsonify({
            'score': int(predicted_score),
            'sentiment': sentiment,
            'cleaned_text': cleaned_review
        })
    except Exception as e:
        print(f"Error: {e}")  # Debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)