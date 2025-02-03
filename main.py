import pandas as pd
path = r"C:\Users\Shushit\OneDrive\Desktop\7th sem pro\Reviews.csv" 
df = pd.read_csv(path)

#step 1: data collection 
#view the first few rows
print(df.head())

#check for missing values
print(df.isnull().sum())

#get basic info
print(df.info())

#view data statistics
print(df.describe())#


# step 2
#drop rows with missing values
df = df.dropna()
#verify if nulls are removed
print(df.isnull().sum())

# drop duplicate rows
df = df.drop_duplicates()
#verify the shape after removing duplicates
print("Dataset after removing duplicates:", df.shape)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|[^\w\s]|\d', '', text)
    tokens = nltk.word_tokenize(text)  # Uses the standard `punkt` tokenizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned_text'] = df['Text'].apply(clean_text)

#view clean dataset
df[['Text', 'cleaned_text']].head()

#keep only useful columns
df = df[['cleaned_text', 'Score']]
df.head()


#step 3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#plot the distribution of review sores
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Score', hue='Score', palette='viridis', legend=False)
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Count')
plt.show()


from wordcloud import WordCloud
#join all review into a single string
all_reviews = ' '.join(df['cleaned_text'])

#generate a woord cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

#display the word cloud
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Frequent Words')
plt.show()


from nltk.sentiment.vader import SentimentIntensityAnalyzer


#initialize vader sentiment analyzer
sia = SentimentIntensityAnalyzer()
#function to get sentiment score 
def get_sentiment(text):
    sentiment= sia.polarity_scores(text)
    return sentiment['compound']

#apply sentiment analysis
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

#show sample with sentiment scores
print(df[['cleaned_text', 'sentiment']].head())

#plot correlation between score and sentiment 
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x= 'sentiment', y='Score', hue='sentiment', palette='coolwarm')
plt.title("Correlation between Review Score and Sentiment")
plt.xlabel('Sentiment Score')
plt.ylabel('Review Score')
plt.show()

# step 4
from sklearn.preprocessing import LabelEncoder

#convert the scores to numerical labels
label_encoder = LabelEncoder()
df['Score'] = label_encoder.fit_transform(df['Score'])

print(df[['cleaned_text', 'Score']].head())


from sklearn.model_selection import train_test_split
x= df['cleaned_text'] #features: cleaned text data
y= df['Score'] #labels: encoded review scores

#split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=43)
x_test,x_train,y_test,y_train

from sklearn.feature_extraction.text import TfidfVectorizer

#create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features= 5000) #limit to 5000 feature for efficiency

#fit and transform training data, transform test data
x_train_tfidf= vectorizer.fit_transform(x_train)
print(x_train_tfidf)
x_test_tfidf = vectorizer.transform(x_test)
x_test_tfidf

from sklearn.linear_model import LogisticRegression

#train a logistic regression model
model = LogisticRegression(max_iter= 1000, class_weight='balanced')
model.fit(x_train_tfidf, y_train)

#Evaluate the model
accuracy = model.score(x_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#binarize the labels for multiclass ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_prob= model.predict_proba(x_test_tfidf)

#plot ROC curve
plt.figure(figsize=(8,6))

#calculate ROC curve and AUC
for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# plt the diagonal (chance level)
plt.plot([0,1],[0,1], color='gray', lw=2, linestyle='--')
#customize the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()
#make prediction 
y_pred= model.predict(x_test_tfidf)

#print the classification report
print("Classification Report: \n", classification_report(y_test,y_pred))

#display the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot= True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Binarize the true labels for multiclass
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Get predicted probabilities
y_pred_prob = model.predict_proba(x_test_tfidf)

# Plot precision-recall curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    # Compute precision-recall curve for each class
    precisions, recalls, thresholds = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
    
    # Compute average precision score
    avg_precision = average_precision_score(y_test_bin[:, i], y_pred_prob[:, i])
    
    # Plotting the curve
    plt.plot(recalls, precisions, lw=2, label=f'Class {i} (AP = {avg_precision:.2f})')

# Set plot labels and title
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multiclass Precision-Recall Curve")
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

import joblib
#save the trained Model
joblib.dump(model, 'review_score_model.pkl')

#save the TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

def predict_reviw_score(review_text):
    #load the model and vectorizer

    model = joblib.load('review_score_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    cleaned_review = clean_text(review_text)

    review_tfidf = vectorizer.transform([cleaned_review])

    predicted_score = model.predict(review_tfidf)

    return predicted_score[0]

sample_review = input('Enter a Review: ')
predicted_score = predict_reviw_score(sample_review)
print(f"Predicted Review Score: {predicted_score}")
