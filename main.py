import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import Model
import matplotlib.pyplot as plt

def prepare_data(df,le):
  
  #preprocess data from str to int
  for i in df.columns[1:9]:
    df[i]=le.fit_transform(df[i])
  y = df['Result']#labels
  X = df[df.columns[0:9]]#data

  return X,y

model = Model()
le = LabelEncoder()

#upload data
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data=pd.read_csv(uploaded_file,sep=";")
    df = data.copy()

    X,y=prepare_data(df,le)

    model.train(X,y)

    d = st.text_input("match")
    d = list(d.split(' '))

    result = model.predict(d,le)

    for i in range(3):
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.axis('equal')
        ax.pie(result[i].tolist()[0], labels=model.model1.classes_,autopct='%1.1f%%',shadow=True, startangle=90)
        st.pyplot(fig)
