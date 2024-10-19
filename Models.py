import os
import pandas
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer         as TFIDF
from sklearn.naive_bayes             import MultinomialNB           as NaiveBayes
from sklearn.svm                     import SVC                     as SVM
from sklearn.ensemble                import RandomForestClassifier  as RandomForest
from sklearn.neighbors               import KNeighborsClassifier    as KNN
from sklearn.linear_model            import LogisticRegression
from sklearn.metrics                 import accuracy_score, confusion_matrix

def getLabel(filename):
    return 'Spam' if filename.startswith('spm') else 'Not Spam'

trainData       = []
testData        = []
root            = os.path.dirname(os.path.abspath(__file__))
trainingPath    = os.path.join(root, 'train-mails')
testingPath     = os.path.join(root, 'test-mails')

iterator        = [     (trainingPath, trainData),
                        (testingPath,  testData)
                    ]

for path,container in iterator:
    for file in os.listdir(path):
        filePath    = os.path.join(path,file)
        content     = open(filePath, 'r', encoding='utf-8', errors='ignore').read()
        label       = getLabel(file)
        container.append({'Content': content, 'Label': label})
print('Data parsing complete')

trainData       = pandas.DataFrame(trainData)
testData        = pandas.DataFrame(testData)

trainingLabels  = trainData['Label']
testingLabels   = testData['Label']

vectorize       = TFIDF()
trainingMails   = vectorize.fit_transform(trainData['Content'])
testingMails    = vectorize.transform(testData['Content'])
pickle.dump(vectorize, open('Models/Vectorize.pkl','wb'))
print('Data vectorization complete')

modelAlgorithms = [     NaiveBayes(alpha = 0.1), 
                        SVM(kernel = 'linear'),
                        RandomForest(n_estimators=500, random_state = 11, n_jobs = -1),
                        KNN(weights= 'distance'),
                        LogisticRegression(solver = 'saga', random_state = 11)
                   ]

bestAlgo        = ''
bestAccuracy    = 0
for algorithm in modelAlgorithms:
    algorithm.fit(trainingMails,trainingLabels)
    pickle.dump(algorithm, open(('Models/'+  f'{algorithm}'.split('(')[0]  +'.pkl'),'wb'))
    predictions = algorithm.predict(testingMails)
    accuracy    = 100*accuracy_score(testingLabels,predictions)
    tn,fp,fn,tp = confusion_matrix(testingLabels, predictions).ravel()
    fpr,fnr     = 100*fp/(fp+tn), 100*fn/(fn+tp) 
    print(f'Metrics for {algorithm}:')
    print(f'\tAccuracy: {accuracy:0.3f}%')
    print(f'\tFalse Positive Rate: {fpr:0.3f}%')
    print(f'\tFalse Negative Rate: {fnr:0.3f}%')
    if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestAlgo     = algorithm
    
print(f'Best Algorithm is {bestAlgo} with an accuracy of {bestAccuracy:0.3f}%')






