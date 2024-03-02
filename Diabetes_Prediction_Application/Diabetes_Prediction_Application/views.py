from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    url = ("data\diabetes.csv")
    data = pd.read_csv(url)

    # Let's try balancing the data
    count_class_0, count_class_1 = data['Outcome'].value_counts()

    # Divide by class
    df_class_0 = data[data['Outcome'] == 0]
    df_class_1 = data[data['Outcome'] == 1]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    data = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='log2', max_depth=6, random_state=40)
    model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
   
    if pred == 1:
        return render(request, 'predict.html',{'result2':"Positive"})
    else: 
        return render(request, 'predict.html', {'result2':"Negative"})
    