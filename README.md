# Sentiment Analysis Model of IMDB Movie Reviews

## Abstract
This project involves analyzing the sentiment of IMDB movie reviews using Natural Language Processing (NLP) techniques. The goal is to train a machine learning model to classify movie reviews as positive or negative based on the text. The data used in this project is the IMDB movie reviews dataset, which contains 50,000 movie reviews labeled as positive or negative. The dataset is available on Kaggle (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Methodology
- Preprocessing: Cleaning, tokenizing, and normalizing the text data. This step involves removing stop words, punctuation, and HTML tags, converting all text to lowercase, and stemming or lemmatizing the words.
-Feature Extraction: Extracting features from the text data using Term Frequency-Inverse Document Frequency (TF-IDF).
- Model Building: Training a Naive Bayes classifier on the preprocessed data using scikit-learn.
- Evaluation: Evaluating the performance of the model using metrics such as accuracy, precision, recall, and F1-score.

## Dependencies
- nltk
- pandas
- scikit-learn
- NLTK
To run the code, make sure you have the necessary libraries installed. You can install them using pip or conda. To run the code, simply execute the Python script in a Python environment such as Jupyter Notebook, Google Colab, or a local Python IDE. The output will be displayed in the console or terminal where you run the code.

## Output
After running the code, the output will include the accuracy, precision, recall, and F1-score of the model. The output will be displayed in the console or terminal where you run the code.

## Conclusion
This project demonstrates how to use NLP techniques to analyze sentiment in text data. By training a machine learning model on labeled data, you can automatically classify text as positive, negative, or neutral, and gain insights into the overall sentiment of the data. This can be useful for various applications such as customer reviews, social media posts, or news articles. This project can be extended to include more sophisticated NLP techniques such as word embeddings, deep learning models, and transfer learning. The model can also be deployed as a web application or integrated into an existing system to analyze sentiment in real-time text data.

## References
IMDB movie reviews dataset: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
NLTK: https://www.nltk.org/
scikit-learn: https://scikit-learn.org/stable/
NLTK Corpora: https://www.nltk.org/nltk_data/
Term Frequency-Inverse Document Frequency: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
Naive Bayes classifier: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
Accuracy, precision, recall, and F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
