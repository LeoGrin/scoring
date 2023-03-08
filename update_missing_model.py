import streamlit as st
from google.cloud import firestore
from google.cloud.firestore import SERVER_TIMESTAMP
from google.oauth2 import service_account

import numpy as np
from src.prompt_generation_utils import generate_examples_from_prompts
import openai
import datetime
import random
import json



# Authenticate to Firestore with the JSON account key.
# chache with st.cache
@st.cache_resource()
def get_firestore_client():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db

@st.cache_resource()
def load_refs():
    db = get_firestore_client()
    prompts_ref = db.collection("prompts")
    completions_ref = db.collection("completions")
    ratings_ref = db.collection("ratings")
    return prompts_ref, completions_ref, ratings_ref

# Get the prompts from Firestore
prompts_ref, completions_ref, ratings_ref = load_refs()

# For all completions, if the model key is missing, add it as "text-davinci-003"

# Get the completions from Firestore
completions = completions_ref.stream()
for completion in completions:
    completion_dict = completion.to_dict()
    if "model" not in completion_dict:
        completion_dict["model"] = "text-davinci-003"
        completions_ref.document(completion.id).set(completion_dict, merge=True)
