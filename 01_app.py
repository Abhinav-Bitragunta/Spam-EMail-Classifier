import os
import pickle
from flask                           import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer         as TFIDF
from sklearn.naive_bayes             import MultinomialNB           as NaiveBayes
#from sklearn.svm                     import SVC                     as SVM
#from sklearn.ensemble                import RandomForestClassifier  as RandomForest
#from sklearn.neighbors               import KNeighborsClassifier    as KNN
#from sklearn.linear_model            import LogisticRegression
#Change the import lines based on which model is being chosen


app = Flask(__name__)

#Since it performed best, I'm choosing Naive Bayes model here, to choose a different one look in /Models (just don't use Vectorize.pkl)
model       = pickle.load(open('Models/MultinomialNB.pkl','rb')) 
vectorize   = pickle.load(open('Models/Vectorize.pkl', 'rb'))

@app.route('/', methods= ['GET', 'POST'])
def home():
    prediction,color= None,None
    if request.method == 'POST':
        contents    = request.form['email']
        contents    = vectorize.transform([contents])
        prediction  = model.predict(contents)[0]
        color       = 'red' if prediction == 'Spam' else 'green'
    return render_template('index.html', prediction=prediction, color = color)

if __name__ == '__main__':
    app.run(debug=True)