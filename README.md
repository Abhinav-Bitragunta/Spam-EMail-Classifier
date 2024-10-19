# Spam E-Mail Classifier
A web application built using HTML, CSS and Flask that uses an AI model trained on the Enron dataset of 'Spam' and 'Not Spam' E-Mails to classify a user input E-Mail as 'Spam' or 'Not Spam'

### About the model
The dataset can be found the train-mails and test-mails directories.
The script used to train the models can be found in **Models.py**, and the models trained along with the TF-IDF vectorizer can be found in the **Models/** directory.
Multiple models were trained on said dataset and achieved the following results:

![Results of different classifier algorithms](https://i.imgur.com/ei2IP4S.png)

Since it achieved the best results, the web-app uses a Naive Bayes model.

### To change the model used
If you wish to change the model used, then make the following changes in app.py:

```py
#from sklearn.naive_bayes             import MultinomialNB           as NaiveBayes
#from sklearn.svm                     import SVC                     as SVM
#from sklearn.ensemble                import RandomForestClassifier  as RandomForest
#from sklearn.neighbors               import KNeighborsClassifier    as KNN
#from sklearn.linear_model            import LogisticRegression
#Uncomment the model which you wish to use
```


