import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(sequence), axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""


st.title("Next Word Predictor")
st.write("Enter the phrase")


user_input = st.text_input("Enter your text:")


max_sequence_len = model.input_shape[1]

if st.button("Predict Next Word"):
    if user_input:
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        if next_word:
            st.success(f"Predicted next word: *{next_word}*")
        else:
            st.warning("Unable to predict, Try something different")
    else:
        st.error("Enter some text to predict the next word.")
