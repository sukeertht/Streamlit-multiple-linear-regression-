import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm

st.image("https://logos-download.com/wp-content/uploads/2016/10/Deloitte_logo_black.png",width=120)
st.title("M&A Index Prediction Model")
st.markdown("The Deloitte M&A Index is a forward-looking indicator that forecasts futureglobal M&A deal volumes and identifies the factors influencing conditions for dealmaking.")

files = st.file_uploader('File upload', type=['csv'],accept_multiple_files=True)
for file in files:
    dataframe = pd.read_csv(file)
    file.seek(0)
    if file.name=="train.csv":
        train=dataframe
    if file.name=="test.csv":
        test=dataframe

if st.checkbox("Preview Dataset"):
    number=st.slider("Select No of Rows",5,train.shape[0])
    st.write(train.head(number))


def correlation():
    # if st.checkbox("Show Correlation plots"):
        x_val=train.columns
        y_val=train.columns
        z_val=train.corr()
        data=[go.Heatmap(x=x_val,y=y_val,z=z_val.values.tolist(),colorscale='Inferno',)]
        layout=go.Layout(title="Correlation Plot",autosize=False,width=900,height=700,
                    yaxis=go.layout.YAxis(automargin=True),
                    xaxis=go.layout.XAxis(automargin=True),
                    )
        fig=go.Figure(data=data,layout=layout)
        st.plotly_chart(fig)

def model():
    # if st.checkbox("Select Model parameters"):
        features=st.multiselect("Select Independent Variables",train.columns,key='0')
        st.write("You selected", len(features), 'variables') 
        label=st.selectbox("Select Dependent Variable",train.columns,key='1')
        X=train[features].values
        y=train[label].values
        if(st.button("Run the Model")):
            X = sm.add_constant(X)
            result = sm.OLS(y,X).fit()  
            st.text(result.summary())
            pred=result.predict(sm.add_constant(test[features].values))
            st.text(pred)

st.sidebar.title("Select the Desired Option")
st.sidebar.text("(Uncheck the box post analysis to clear)")
if st.sidebar.checkbox("Correlation Plot"):
    correlation()
if st.sidebar.checkbox("Custom Regression Model"):
    model()
