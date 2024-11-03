import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

OPENAI_KEY = st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_KEY)

# This is the function for calling LLM
def get_completion(prompt,
                   model="gpt-4o-mini",
                   temperature=0,
                   top_p=1.0,
                   max_tokens=256,
                   n=1,
                   json_output=False):

    # To check if there is an output in json object format
    if json_output == True:
      output_json_structure = {"type": "json_object"}
    else:
      output_json_structure = None

    # define messages
    messages = [{"role": "user", "content": prompt}]

    # Get LLM response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        response_format=output_json_structure,
    )
    # print(response)
    return response.choices[0].message.content

# This function directly take in "messages" as the parameter.
def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1
    )
    return response.choices[0].message.content

# Functions for Counting Tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))

def count_tokens_from_message(messages):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    value = ' '.join([x.get('content') for x in messages])
    return len(encoding.encode(value))

# Function for Generating Embedding
def get_embedding(input, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return [x.embedding for x in response.data]
