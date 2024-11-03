import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

OPENAI_KEY = st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_KEY)
