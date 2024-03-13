# CAPSTONE PROJECT

# NLP APPLICATIONS

# Python program that performs sentiment analysis

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


# load spaCy language model
# Add the spacytextblob (sentiment) component to the pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

# Read the amazon_product_reviews file
custom_rev_df = pd.read_csv("amazon_product_reviews.csv", 
                            sep= ',', low_memory = False)

# Overview of the dataset
print(custom_rev_df.head())
print(custom_rev_df["reviews.text"].info())
print()

# Select the review.text column and remove missing values
reviews_data = custom_rev_df["reviews.text"].dropna()

# Function to preprocess the data:
# (tokenization, lemmatization, stop_words and ponctuation removal)
def preprocess(text):
    doc = nlp(text)
    return  ' '.join([token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct])
    

# Funtion for sentiment analysis
def pred_sentiments (review):

    review = nlp(review)
    sentiment = review._.blob.sentiment
    polarity = review._.blob.polarity
    #assessment = review._.blob.sentiment_assessments.assessments
    
    # Define sentiment labels
    if polarity > 0:
        sentiment_label = "Positive"
    elif polarity < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    print(f"Review: {review}")
    print(f"Sentiment: {sentiment_label}; {sentiment}")
    #print(assessment)
    print("**"*50)
    return sentiment_label, sentiment

# Test model on a random sample of product reviews
clean_sample_rev_data = reviews_data.sample(200,
                                 random_state=34).apply(preprocess)
clean_sample_rev_data = clean_sample_rev_data.apply(pred_sentiments)

print()
print("-------------reviews similarity---------------")
print()

# Function for similarity analyis 
def simil_compare(*review):
    reviews = [*review]
    for review1 in reviews:
        for review2 in reviews:
            if review1.similarity(review2) == 1 or review1 == review2:
                break
            print(review1,"<->", review2, ":", review1.similarity(review2))
    return None

# Load spaCy medium language model that includes word vectors
nlp = spacy.load("en_core_web_md")

# Process the data with spacy medium language model
clean_rev_data = reviews_data.sample(200, random_state=34).apply(nlp)

# Print reviews index to choose from
print()
print(clean_rev_data.index)

# We can also choose from rewiews with various sentiment_label reviews:
print()
print(clean_sample_rev_data.sample(50, random_state= 5))
print()

# Compare the similarity of reviews (2 positive, 1 negative and 1 neutral review)
simil_compare(clean_rev_data[6857], clean_rev_data[31497],
               clean_rev_data[15930], clean_rev_data[20410])