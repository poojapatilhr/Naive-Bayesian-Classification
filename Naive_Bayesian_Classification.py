

# ===================================
# Naive Bayesian Classification
# =================================== 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, datasets
 # Read the wine data 
wine = datasets.load_wine() 
# Data exploration
wine.keys()
print(wine.DESCR) 
# Assign feature data to X and target data to y
X = wine.data
y = wine.target 
# Check the size
X.shape 
y.shape 
# Partition the data into subsets: one for training and the other for testing, you can set the 
#parameter values differently from the default ones. Set the 'stratify' option 'y' to sample about the 
#equal number of data from ecah wine type.  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.2,random_state=42, stratify=y)
 # Create a Bayesian Classifier instance for classification
gnb = GaussianNB()
 
# Build a Bayesian Classification Model and predict the type using the test data.
gnb.fit(X_train, y_train)
 
# Calculate the posteriori probabilities
p = gnb.predict_proba(X_test)
 
# Predict the target value using the test data.
y_pred = gnb.predict(X_test)
 
# Calculate the accuracy
accuracy = gnb.score(X_test, y_test)
 
# This is the old formatting method.
# print('Accuracy: {:.4f}'.format(accuracy))
print(f'Accuracy: {accuracy: .4f}')
 
# Build a confusion matrix and calculate evaluation ratios
cm = metrics.confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))
 
