
# Spam E-Mail Classifier

A web application built using HTML, CSS and Flask that uses an AI model trained on the [Enron dataset](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html) of 'Spam' and 'Not Spam' E-Mails to classify a user input E-Mail as 'Spam' or 'Not Spam'.

It is currently being hosted on Render, [here](https://spam-email-classifier-qy4o.onrender.com).

  

## About the model

The script used to train the models can be found in **Models.py**, and the models trained(except RandomForest, due to its large size) along with the TF-IDF vectorizer can be found in the **Models/** directory.

Multiple models were trained on this dataset and achieved the following results:

  

![Results of different classifier algorithms](https://i.imgur.com/ei2IP4S.png)

  

Since it achieved the best results, the web-app uses a Naive Bayes model.

  

## To change the model used

If you wish to change the model used, then make the following changes in app.py:

  

```py

#from sklearn.naive_bayes import MultinomialNB as NaiveBayes

#from sklearn.svm import SVC as SVM

#from sklearn.ensemble import RandomForestClassifier as RandomForest

#from sklearn.neighbors import KNeighborsClassifier as KNN

#from sklearn.linear_model import LogisticRegression

#Uncomment the model which you wish to use

```

and

  

```py

#Replace MultinomialNB with a filename in the Models/ directory, anything except Vectorize

model = pickle.load(open('Models/MultinomialNB.pkl','rb'))

```

## To retrain the models

1. Download the .tar.gz files from the drive link in static/test_train_mails.txt
2. Extract them into appropriately named folders(**'test-mails'** and **'train-mails'**)
3. Paste these folders into the root directory of the repository
4. Run Models.py after changing the parameters as you wish

## Navigating the repository

### 1. Models/

This directory contains the pretrained models and the vectorizer used to transform the data as .pkl files.

The names specify the algorithm used to train them.

### 2. static/

This directory contains the CSS code for the web-app

### 3. static/test_train_mails.txt

Containts drive link to dataset needed to retrain the models.

### 4. templates/

This directory contains the HTML file for the web-app

### 5. Models.py

This python script contains the code used to train each model and store them in Models/.

It also determines the best one among them based on accuracy.

### 6. app.py

This python script is the back-end of the web-app, that uses Flask to fetch from and send data to it.

It loads the Naive Bayes model and the appropriate vectorizer to classify the user input as either 'Spam' or 'Not Spam'

