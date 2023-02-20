import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load All Files

with open('best_tree.pkl', 'rb') as file_model:
  model = pickle.load(file_model)



def run():
    st.title('CC Default Prediction')

    # Membuat form
    with st.form(key='form_parameters'):

        limit_balance = st.number_input('Limit Balance', min_value=0, max_value=None, value= 130000)
        sex = st.selectbox('Sex', (1, 2), index=1, help='Gender of the Customers, 1 Male, 2 Female.')
        education_level= st.selectbox('Education Level', (0, 1, 2, 3, 4, 5, 6), index=1, help='Education Level of the Customers.')
        marital_status= st.selectbox('Marital Status', (0, 1, 2, 3), index=1, help='Marital Status of the Customers.')
        age = st.number_input('Age', min_value=0, max_value=100, value= 45)
        pay_0 = st.number_input('Pay 0', min_value=-2, max_value=2, value= 0)
        pay_2 = st.number_input('Pay 2', min_value=-2, max_value=2, value= 0)
        pay_3 = st.number_input('Pay 3', min_value=-2, max_value=2, value= 0)
        pay_4 = st.number_input('Pay 4', min_value=-2, max_value=2, value= 0)
        pay_5 = st.number_input('Pay 5', min_value=-2, max_value=2, value= 0)
        pay_6 = st.number_input('Pay 6', min_value=-2, max_value=2, value= 0)
        bill_amt_1 = st.number_input('Bill amt 1', min_value=0, value= 58180)
        bill_amt_2 = st.number_input('Bill amt 2', min_value=0, value= 59134)
        bill_amt_3 = st.number_input('Bill amt 3', min_value=0, value= 61156)
        bill_amt_4 = st.number_input('Bill amt 4', min_value=0, value= 62377)
        bill_amt_5 = st.number_input('Bill amt 5', min_value=0, value= 63832)
        bill_amt_6 = st.number_input('Bill amt 6', min_value=0, value= 65099)
        pay_amt_1 = st.number_input('Pay amt 1', min_value=0, value= 2886)
        pay_amt_2 = st.number_input('Pay amt 2', min_value=0, value= 2908)
        pay_amt_3 = st.number_input('Pay amt 3', min_value=0, value= 2129)
        pay_amt_4 = st.number_input('Pay amt 4', min_value=0, value= 2354)
        pay_amt_5 = st.number_input('Pay amt 5', min_value=0, value= 2366)
        pay_amt_6 = st.number_input('Pay amt 6', min_value=0, value= 2291)
        
        
        submitted = st.form_submit_button('Predict')

    data_inf = {
      'limit_balance' : limit_balance,
      'sex' : sex,
      'education_level': education_level,
      'marital_status': marital_status,
      'age' : age,
      'pay_0' : pay_0,
      'pay_2' : pay_2,
      'pay_3' : pay_3,
      'pay_4' : pay_4,
      'pay_5' : pay_5,
      'pay_6' : pay_6,
      'bill_amt_1' : bill_amt_1,
      'bill_amt_2' : bill_amt_2,
      'bill_amt_3' : bill_amt_3,
      'bill_amt_4' : bill_amt_4,
      'bill_amt_5' : bill_amt_5,
      'bill_amt_6' : bill_amt_6,
      'pay_amt_1' : pay_amt_1,
      'pay_amt_2' : pay_amt_2,
      'pay_amt_3' : pay_amt_3,
      'pay_amt_4' : pay_amt_4,
      'pay_amt_5' : pay_amt_5,
      'pay_amt_6' : pay_amt_6
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)
    

    if submitted:       
        pred = model.predict(data_inf)
        st.write('Hasil Prediksi : ', pred[0])

      
if __name__ == '__main__':
    run()