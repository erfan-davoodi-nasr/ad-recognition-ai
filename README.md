# ad-recognition-ai
Recognition of texts containing advertisements in English with Python and the scikit-learn library
The provided code is an example of a simple text classification model using the scikit-learn library in Python. This model is designed to classify text data into two categories: advertisements (label 1) and non-advertisements (label 0). Here's a detailed explanation of the code:

    Importing necessary libraries:
        CountVectorizer from sklearn.feature_extraction.text is used to convert text data into a numerical format suitable for machine learning.
        MultinomialNB from sklearn.naive_bayes is a Naive Bayes classifier used for text classification.
        train_test_split from sklearn.model_selection is used to split the dataset into training and testing sets.
        accuracy_score from sklearn.metrics is used to evaluate the accuracy of the model.

    Data Preparation:
        texts is a list containing sample text messages.
        labels is a corresponding list of labels (1 for advertisements, 0 for non-advertisements).

    Vectorization:
        CountVectorizer is used to convert the text data into a numerical format. It creates a matrix where each row represents a document (text message), and each column represents a unique word in the entire corpus. The values in the matrix represent the word frequencies in each document.

    Data Splitting:
        The dataset is split into training and testing sets using the train_test_split function. The training set is used to train the model, and the testing set is used to evaluate its performance.

    Model Training:
        A Multinomial Naive Bayes classifier is created using MultinomialNB().
        The model is trained on the training data using model.fit(features_train, labels_train).

    Prediction:
        The code allows you to input a new text for classification.
        The input text is transformed into numerical features using the same CountVectorizer object used for training.
        The model predicts the label (1 for an advertisement or 0 for a non-advertisement) for the input text.

    Model Evaluation:
        The accuracy of the model is calculated by comparing the predicted labels on the testing set with the true labels, and it is printed as a percentage.

The primary goal of this code is to classify text messages as advertisements or non-advertisements based on the content of the messages. It's a basic example of text classification using a simple Naive Bayes model. You can adapt this code for your specific text classification tasks by providing your own dataset and labels.
