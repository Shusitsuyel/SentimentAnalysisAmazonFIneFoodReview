# SentimentAnalysisAmazonFIneFoodReview

#Project Overview
This web application uses sentiment analysis to predict whether the sentiment of given Amazon Fine Food Review is positive. neutral, negative, or very negative. Built with Flask, it leverage machine learning techniques including TF-IDF vectorization and Natural Language Processing (NLP) to classify the sentiment of reviews. This project is a part of a group initiative, demonstrating a practical application of NLP for real-world data analysis.

#Key Features
>> Simple User Interface : The app allows users to input a review and get sentiment prediction instantly.
>> Sentiment prediction : Allows users to input a food review and predict its sentiment (Positive or negative).
>> Machine Learning Integration : Uses a Pre-Trained model saved with 'joblib' to make predictions.
>> Text Preprocessing and TF-IDF : The app uses NLTK for text preprocessing(tokenization, stopword removal, stemming) and Scikit-learn's TF-IDF vectorizer for feature extraction.

#Tech Stack
The project leverage the following technologies and tools:
>> Backend:
  > Flask : A python web framework used to build the web application. It allows us to handle HTTP requests and route users to different pages within the app.
>> NLP Libraries:
  > NLTK(Natural Language Toolkit) : A library for working with human language data. It is used for text preprocessing, such as tokenization and stopword removal
  > Scikit-Learn(sklearn) : A machine learning library that provides the tools for building the TF-IDF vectorizer and classifier model.
>> Machine Learning Model: The machine learning model is a classifer that predict the sentiment of the review text. It uses:
  > TF-IDF(Term Frequency-Inverse Document Frequency) : A vectorization technique that transforms the textual data into numeriacal format for machine learning.
>> Model Serialization:
  > joblib : used to save and load the trained machine learning model. This enables the model to be reused for prediction without retraining each time.

