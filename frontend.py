import random
import time
import streamlit as st
from chatbot_BART import ChatBot
from setup_qdrant import setup_qdrant
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from pprint import pprint
import torch

filename = "bigBasketProducts.csv"
model_name = "vblagoje/bart_lfqa"

st.title("Basket Bot")

if "indices_dict" not in st.session_state:
    # st.write("Setting up Qdrant for queries, this will take 2-3 minutes, please wait...")
    with st.spinner('Setting up Qdrant for queries, this will take 2-3 minutes, please wait...'):
        indices_dict = setup_qdrant(6333, 'bigBasketProducts.csv')
    st.session_state["indices_dict"] = indices_dict
else :
    indices_dict = st.session_state["indices_dict"]

if "bot" not in st.session_state:
    # st.write("Loading model, this will take a minute, please wait...")
    with st.spinner('Loading model, this will take a minute, please wait...'):
        bot = ChatBot(model_name, filename, indices_dict)
    st.session_state["bot"] = bot
else:
    bot = st.session_state["bot"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

def write_response(assistant_response, message_placeholder):
    full_response = ""
    for chunk in assistant_response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)

# Display assistant response in chat message container

with st.chat_message("assistant"):
    message_placeholder = st.empty()
    response = bot.get_response(prompt)
    write_response(response, message_placeholder)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    