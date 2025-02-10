import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequences import pad_sequences

model = load_model ("next_word_lstm.h5")

with open("tokenizer.pkl",'rb') as handle:
    tokenizer=pickle.load(handle)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None  # Return None if word not found in tokenizer

st.title("Next word predict")
input_text=st.text_input("Enetr the sequence of word", "The sun is shining ")
if st.button("Predict: Next word"):
    max_sequences_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequences_len)
    st.write('the predicition next word is: ',next_word) 