import streamlit as st
import time
import numpy as np
import pandas as pd
import json

from classifier import BackgroundModel


st.title('My first app')


model = BackgroundModel("dummy")
model_classes = [model.classes[i].name for i in range(len(model.classes))]


with st.beta_expander("Save/Load"):
    chart_data = pd.DataFrame(
        np.random.randn(1, len(model_classes)),
        columns=model_classes)
    st.bar_chart(chart_data)
    load = st.button("Load Data")
    load = st.button("Save Data")

with st.beta_expander("Training"):
    enable_training = st.button("Train Classes")
    enable_show_traing = st.button("Show Classes")
    class_type = st.radio(
        "class_type",model_classes)


with st.beta_expander("Border Adjustment"):
    enable_adjust = st.button("Enable Adjustment")
    adjust_up = st.button("Up")


# Data to be written
dictionary ={
    "state" : "blah",
    "rollno" : 56,
    "cgpa" : 8.6,
    "phonenumber" : "9976770500"
}
  
# Serializing json 
json_object = json.dumps(dictionary, indent = 4)
  
# Writing to sample.json
with open("app_state.json", "w") as outfile:
    outfile.write(json_object)