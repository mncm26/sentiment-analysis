# sentiment-analysis
Python code defining a function to analyse the sentiment of Amazon product reviews
“Consumer Reviews of Amazon Products” was downloaded from Kaggle.com. The dataset has 34,660 reviews, ratings, information… about electronic Amazon products (Echo, Fire Stick, Kindle tablet…) provided by Datafiniti’s Product  Database. 
Only the column “reviews.text” is needed for this program

Read the (renamed) amazon_product_reviews.csv.
Create a data frame.
Select the column needed and retrieve its data.

Preprocessing the data:

Remove the null values using dropna().
Tokenize the sentences using .nlp().
Remove capital letters using .lower().
Lemmatize the words using .lemma_.
Remove stop words using .is_stop.
Remove punctuation using .is_punct.

The sentiment analysis function is designed to take a product review as input and return its sentiment label. The function is tested on a random sample of 200 preprocessed reviews from the dataset to evaluate its performance on various reviews. The polarity (and subjectivity) score is calculated and a sentiment label is successfully assigned to each review providing insights into the sentiment analysis results. 

Thanks to spaCy and the pre-trained textblob library, the model effectively and efficiently outputs a high percentage of accurate sentiment labels. For further insights, the function can output polarity and subjectivity scores for each assessed token in the text using ._.sentitment_assessments.assessments. 
However, the sentiment intensity may not be very accurate: texblob is trained on general-purpose text data. In this instance, for reviews on electronic products, some words such as ‘game’, ‘remote’, and ‘small’ … are negatively polarised in the textblob library but are rarely/never used negatively in the reviews. The use of those words reduces the positive sentiment intensity or, in some cases, wrongly labels the sentiment. 
For more accuracy,  the text also requires preprocessing in addition to NLP which means higher processing times.
