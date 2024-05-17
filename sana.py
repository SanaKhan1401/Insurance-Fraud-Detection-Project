import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
#----avinash---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle



def main():
    st.title("Insurance Fraud Detection Project")
    st.sidebar.title("Insurance Fraud Detection Project")
    st.markdown("Lets Detect Fraud")
    st.sidebar.markdown("Lets Detect Fraud")

#---- load data---
    st.cache(persist=True)
    def load_data():
        data=pd.read_csv("[Dataset]_Module8_(Insurance).csv",dtype=float)
        return data

    st.cache(persist=True)
    def split(df):
        x = df.drop(['fraud_reported','Unnamed: 0'], axis = 1)
        y = df['fraud_reported'].map({0:"No Fraud",1:"Yes Fraud"})
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test
    
    df=load_data()

    x_train,x_test,y_train,y_test = split(df)
    classifier=st.sidebar.selectbox("Classifier",("DCT","RF","KNN","SVC"))

    # DTC Hyper Parameters Set
    if classifier=="DCT":
        st.sidebar.subheader("HyperParameters")
        max_depth =st.sidebar.number_input("max_depth",1,100,step=1,key="max_depth")
        min_samples_split = st.sidebar.number_input("min_samples_split",1,1000,step=1,key="min_samples_split")
        min_samples_leaf=st.sidebar.number_input("min_samples_leaf",1,100,step=1,key="min_samples_leaf")
    # Train classifier
    

        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("Decision Tree Classifier")
            dtc = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
            dtc.fit(x_train,y_train)
            y_pred=dtc.predict(x_test)
            acc=accuracy_score(y_pred,y_test)
            result=dtc.predict(x_test)
            st.write("Accuracy:",acc.round(2))
            st.write("Input Parameters",x_test)
            st.write("Fraud Report",result)
            #st.write("Precision",precision_score(y_test,y_pred).round(2))
            #st.write("Recall",recall_score(y_test,y_pred).round(2))
            
    # Random Forest 
    if classifier=="RF":
        st.sidebar.subheader("HyperParameters")
        max_depth =st.sidebar.number_input("max_depth",1,100,step=1,key="max_depth")
        min_samples_split = st.sidebar.number_input("min_samples_split",1,1000,step=1,key="min_samples_split")
        min_samples_leaf=st.sidebar.number_input("min_samples_leaf",1,100,step=1,key="min_samples_leaf")
        n_estimators=st.sidebar.number_input("n_estimators",1,1000,key="n_estimators")
    # Train classifier
       

        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("Random Forest Classifier")
            rf= RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
            rf.fit(x_train,y_train)
            y_pred=rf.predict(x_test)
            acc=accuracy_score(y_pred,y_test)
            result=rf.predict(x_test)
            st.write("Accuracy:",acc.round(2))
            st.write("Input Parameters",x_test)
            st.write("Fraud Report",result)
            
            #st.write("Precision",precision_score(y_test,y_pred).round(2))
            #st.write("Recall",recall_score(y_test,y_pred).round(2))
            
    # KNN
    if classifier=="KNN":
        st.sidebar.subheader("HyperParameters")
        n_neighbors =st.sidebar.number_input("n_neighbors",1,100,step=1,key="n_neighbors")
       
    # Train classifier
       

        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("K Nearest Neghbors")
            knn = KNeighborsClassifier(n_neighbors = n_neighbors)
            knn.fit(x_train,y_train)
            y_pred=knn.predict(x_test)
            acc=accuracy_score(y_pred,y_test)
            result=knn.predict(x_test)
            st.write("Accuracy:",acc.round(2))
            st.write("Input Parameters",x_test)
            st.write("Fraud Report",result)
            
            #st.write("Precision",precision_score(y_test,y_pred).round(2))
            #st.write("Recall",recall_score(y_test,y_pred).round(2))

     # SVC
    if classifier=="SVC":
        st.sidebar.subheader("HyperParameters")
        st.sidebar.subheader("No Hyper Parameters are passed")
       
    # Train classifier
       

        if st.sidebar.button("Classify",key="Classify"):
            st.subheader("Support Vector Classifier")
            svm = SVC()
            svm.fit(x_train,y_train)
            y_pred=svm.predict(x_test)
            result=svm.predict(x_test)
            acc=accuracy_score(y_pred,y_test)
            st.write("Accuracy:",acc.round(2))
            st.write("Input Parameters",x_test)
            st.write("Fraud Report",result)
            
    
    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("Insurance Fraud Detected")
        st.write(df)

    

#---Done --
    st.markdown("Developed by External Guide Avinash Pawar and WBL intern : Sana Khan at NIELIT Daman")
   

    



if __name__ == '__main__':
    main()


