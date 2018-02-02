#dependencies
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import itertools
import matplotlib.pyplot as plt
import pandas as pd


#function defination to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


train_data = pd.read_csv('criminal_train.csv')
y_train = train_data['Criminal']
X_train = train_data.drop(['PERID','Criminal'],axis = 1).values
X_train = normalize(X_train, axis = 0)

test_data = pd.read_csv('criminal_test.csv')
X_test = test_data.drop(['PERID'],axis = 1).values
X_test = normalize(X_test, axis = 0)


#model structure
model = VotingClassifier(
    estimators=[
                ( 'gb',GradientBoostingClassifier(n_estimators=500,verbose =1,max_depth = 6 )), 
                ('rf', RandomForestClassifier(n_estimators=1000, verbose = 1))], 
    voting='soft') 

model = AdaBoostClassifier(base_estimator= model, n_estimators =10 )
#training the model
print('training the model: ')
model.fit(X_train, y_train)
print('model trained: ')
model.score(X_train,y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train ,y_train, train_size = .9)
model.fit(X_train, y_train)
model.score(X_test, y_test)
#####################################################
#predicting values on test file
df = pd.read_csv('criminal_test.csv')
predicted = model.predict(X_test)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_old.csv', index = False)
###############################################################

print(model.score(X_train, y_train))
# predicted value 
predicted_y = model.predict(X_train)

#plot the confusion matrix
cnf_matrix = confusion_matrix(y_train, predicted_y)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

################################################3
#input file
df = pd.read_csv('criminal_train.csv')
#    KNN classifier
model_knn =KNeighborsClassifier(n_neighbors= 5, weights='distance', n_jobs = 4)
model_knn.fit(X_train, y_train)
model_knn.score(X_train,y_train)
predicted = model_knn.predict(X_train)
df = pd.read_csv('criminal_train.csv')
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('input_knn.csv', index = False)

## random forest classifier
model_rf =RandomForestClassifier(n_estimators=1000, verbose = 1)
model_rf.fit(X_train, y_train)
df = pd.read_csv('criminal_train.csv')
predicted = model_rf.predict(X_train)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('input_rf.csv', index = False)


# ada boosting clssifier
model_ab = AdaBoostClassifier(n_estimators=500)
model_ab.fit(X_train, y_train)
df = pd.read_csv('criminal_train.csv')
X_test = df.drop(['PERID'],axis =1).values
predicted = model_ab.predict(X_train)

frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('input_ab.csv', index = False)

### gradient boosting classifier
model_gb = GradientBoostingClassifier(n_estimators=500,verbose =1,max_depth = 6 )
model_gb.fit(X_train, y_train)

df = pd.read_csv('criminal_train.csv')
X_test = df.drop(['PERID'],axis =1).values
predicted = model_gb.predict(X_train)

frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('input_gb.csv', index = False)

#logistic regression
model_lr  =LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=.3)
model_lr.fit(X_train, y_train)
predicted = model_lr.predict(X_train)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('input_lr.csv', index = False)

## support vector machines
model_svm  =svm.SVC(C=.75, verbose = True)
model_svm.fit(X_train, y_train)
predicted = model_svm.predict(X_train)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('input_svm.csv', index = False)


##############################################
### output file

test_data = pd.read_csv('criminal_test.csv')
X_test = test_data.drop(['PERID'],axis = 1).values
X_test = normalize(X_test, axis = 0)

#    KNN classifier
predicted = model_knn.predict(X_test)
df = pd.read_csv('criminal_test.csv')
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_knn.csv', index = False)

## random forest classifier
predicted = model_rf.predict(X_test)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_rf.csv', index = False)


# ada boosting clssifier
predicted = model_ab.predict(X_test)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_ab.csv', index = False)

### gradient boosting classifier
predicted = model_gb.predict(X_test)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_gb.csv', index = False)


### gradient logistic regression classifier
predicted = model_lr.predict(X_test)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_lr.csv', index = False)

## support vector machines
predicted = model_svm.predict(X_test)
frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output_svm.csv', index = False)


####################################################
### WRITING THE ACCURACY
print('Knn score: ', model_knn.score(X_train, y_train))
print('gaussian nb score: ', model_gb.score(X_train, y_train))
print('ada boost score: ', model_ab.score(X_train, y_train))
print('random forest score: ', model_rf.score(X_train, y_train))
print('logistic regression: ',model_lr.score(X_train, y_train))
print('support vector machines:',model_svm.score(X_train, y_train))
##############################################################
df = pd.read_csv('criminal_test.csv')
test_data = pd.read_csv('criminal_test.csv')
X_test = test_data.drop(['PERID'],axis = 1).values
X_test = normalize(X_test, axis = 0)
predicted = model.predict(X_test)

frame = pd.DataFrame()
frame['PERID'] = df['PERID']
frame['Criminal'] = predicted
frame.to_csv('output.csv', index = False)


