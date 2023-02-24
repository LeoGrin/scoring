import streamlit as st
import pandas as pd
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
#@st.cache(allow_output_mutation=True) #TODO cache for read quota but change how we write new things
def get_firestore_client():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    return db

# Get the prompts from Firestore
db = get_firestore_client()
print("database loaded")
prompt_ref = db.collection("prompts")

openai.api_key = st.secrets["API_KEY"]


# Who are you?
user = st.sidebar.text_input("Who are you?", value="anonymous")

model = st.sidebar.selectbox("Select a model", options=["text-davinci-003"])

# Add a selectbox to select a prompt
# get brands from prompt_ref
# Get all the documents in the collection and get the brand attribute
# @st.cache(allow_output_mutation=True)
# def load_brands():
#     print("Loading brands..")
#     brand_docs = db.collection('brands').get()
#     brands = [doc.id for doc in brand_docs]
#     print(brands)
#     return brands

# brands = load_brands() #FIXME
brands = ["Frichti"]


brand = st.sidebar.selectbox("Select a brand", options=brands)

# Select a min date
min_date = st.sidebar.date_input("Select a min date", value=datetime.date(2021, 1, 1))
# convert to number
min_date = datetime.datetime.strptime(min_date.strftime("%d/%m/%Y"), "%d/%m/%Y").date()
#Select a prompt created after the min date
#prompts = [doc.to_dict()["prompt"] for doc in prompt_ref.stream() if doc.to_dict()["brand"] == brand and datetime.datetime.strptime(doc.to_dict()["date"], "%d/%m/%Y %H:%M:%S").date() >= min_date]
@st.cache(allow_output_mutation=True)
def load_prompts(brand, min_date):
    print("Loading prompts..")
    prompts = []
    prompt_ids = []
    query = prompt_ref.where("brand", "==", brand).where("date", ">=", datetime.datetime.combine(min_date, datetime.time.min))
    for doc in query.stream():
        prompt = doc.to_dict()["prompt"]
        prompt_id = doc.id
        prompts.append(prompt)
        prompt_ids.append(prompt_id)
    print("Done!")
    return prompts, prompt_ids

prompts, prompt_ids = load_prompts(brand, min_date)

# Choose with radio buttons if you want to add a new prompt or look at the completions
add_prompt = st.sidebar.radio("Add a new prompt or look at the completions", options=["Look at the completions", "Add a new prompt"])
if add_prompt == "Add a new prompt":
    # Create a new prompt
    # Create a form
    with st.form("add_prompt"):
        # Add a text input to enter the new prompt
        # make it multiline
        new_prompt = st.text_area("Enter the new prompt", height=200)
        # Add a submit button
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            st.balloons()
            print(f"Adding a new prompt: {new_prompt} for brand {brand} by user {user}")
            # Add the new prompt to the database
            prompt_ref.add({"brand": brand, "prompt": new_prompt, "user": user,
                        "date": SERVER_TIMESTAMP})
            st.experimental_rerun()
else:
    if len(prompts) == 0:
        st.warning("No prompt for this brand and this date")
    else:
        prompt = st.sidebar.selectbox("Select a prompt", options=prompts)
        # Display the completions as a rich text with the possibilty to rate them
        # get current prompt id
        st.header(f"Full prompt")
        with st.expander("Show prompt"):
            st.text(prompt)
        # get current prompt id
        prompt_id = [prompt_ids[i] for i, prompt_ in enumerate(prompts) if prompt_ == prompt][0]
        print("prompt_id", prompt_id)
        # get completions from prompt id
        completion_ref = db.collection("completions")
        ratings_ref = db.collection("ratings")
        #completions = [doc.to_dict()["completion"] for doc in completion_ref.stream() if doc.to_dict()["prompt_id"] == prompt_id]
        completions = []
        query = completion_ref.where("prompt_id", "==", prompt_id)
        for doc in query.stream():
            completion = doc.to_dict()["completion"]
            completions.append(completion)
        st.header(f"{len(completions)} completions")
        # Add the possibility to add new completions
        # create a form to add a new completion
        st.header("Add new completions")
        with st.form("add_completion"):
            # umber input to select the number of completions to add
            n_completions = st.number_input("Number of completions to add (⚠️ expensive)", min_value=1, max_value=100, value=1, key="n_completions")
            # Generation parameters
            with st.expander("Generation parameters"):
                # max_tokens
                max_tokens = st.number_input("Max tokens", min_value=1, max_value=500, value=100, key="max_tokens")
                # temperature
                temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.7, key="temperature")
                # top_p
                top_p = st.number_input("Top_p", min_value=0.0, max_value=1.0, value=1.0, key="top_p")
                # frequency_penalty
                frequency_penalty = st.number_input("Frequency penalty", min_value=0.0, max_value=1.0, value=0.0, key="frequency_penalty")
                # presence_penalty
                presence_penalty = st.number_input("Presence penalty", min_value=0.0, max_value=1.0, value=0.0, key="presence_penalty")
                # stop sequence
                stop_sequence = st.text_input("Stop sequence", value="END", key="stop_sequence")
            # submit button
            submit_button = st.form_submit_button("Submit")
            if submit_button:
                with st.spinner(f"Generating {n_completions} completions..."):
                    # Generate the completions
                    response = generate_examples_from_prompts(prompt, 
                                                                n_examples_per_prompt=n_completions, 
                                                                engine=model, 
                                                                max_tokens=int(max_tokens),
                                                                temperature=temperature, 
                                                                top_p = top_p, 
                                                                frequency_penalty = frequency_penalty, 
                                                                presence_penalty = presence_penalty, 
                                                                stop = [stop_sequence], 
                                                                check_if_truncated=True)
                    completions, truncated_list = zip(*response)
                    print(f"Truncated list: {truncated_list}")
                # Add the completions to the database
                for i, completion in enumerate(completions):
                    truncated = truncated_list[i]
                    completion_ref.add({"prompt_id": prompt_id, "completion": completion, "truncated": truncated, "user": user, "model": model, 
                                        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "frequency_penalty": frequency_penalty, "presence_penalty": presence_penalty, "stop_sequence": stop_sequence,
                                        "date": SERVER_TIMESTAMP})
                st.balloons()
                st.experimental_rerun()
        
        st.header("Score the completions")
        # Checkbox to display the existing ratings
        display_ratings = st.checkbox("Display the existing ratings")
        # Checkbox to hide truncated completions
        hide_truncated = st.checkbox("Hide truncated completions")
        # Radio to sort by rating, by date, or randomly
        sort_by = st.radio("Sort by", options=["Rating", "Date", "Random"])
        # Get completions ids
        #completion_ids = [doc.id for doc in completion_ref.stream() if doc.to_dict()["prompt_id"] == prompt_id]
        completion_ids = []
        query = completion_ref.where("prompt_id", "==", prompt_id)
        for doc in query.stream():
            completion_id = doc.id
            completion_ids.append(completion_id)
        if sort_by == "Rating":
            # Get the completions and sort them by rating
            completions_ids = sorted(completion_ids, key=lambda completion_id: np.mean([doc.to_dict()["rating"] for doc in ratings_ref.stream() if doc.to_dict()["completion_id"] == completion_id]), reverse=True)
            # print mean ratings
            #mean_ratings = [np.mean([doc.to_dict()["rating"] for doc in ratings_ref.stream() if doc.to_dict()["completion_id"] == completion_id]) for completion_id in completion_ids]
            #print(np.sort(mean_ratings, reverse=True))
        elif sort_by == "Date":
            # Get the completions and sort them by date
            # check if the key in completion_ref is equal to completion_id
            completions_ids = sorted(completion_ids, key=lambda completion_id: [doc.to_dict()["date"] for doc in completion_ref.stream() if doc.id == completion_id][0], reverse=True)
        elif sort_by == "Random":
            # Get the completions and sort them randomly
            completions_ids = sorted(completion_ids, key=lambda completion_id: random.random(), reverse=True)
        # Loop over the completions
        for i, completion_id in enumerate(completions_ids):
            # Get the completion
            #completion = [doc.to_dict()["completion"] for doc in completion_ref.stream() if doc.id == completion_id][0]
            # query the completion_ref to get the completion
            completion_doc = completion_ref.document(completion_id).get()
            if completion_doc.exists:
                completion = completion_doc.to_dict()["completion"]
                date = completion_doc.to_dict()["date"]
                if "truncated" in completion_doc.to_dict():
                    truncated = str(completion_doc.to_dict()["truncated"])
                else:
                    truncated = "Unknown"
            else:
                completion = None
            # Write the number of completions
            # create a box to display the completion
            if not hide_truncated or truncated == "False":
                with st.expander(f"Completion {i}, created: {date}, truncated: {truncated}", expanded=True):
                    # Write the completion
                    st.write(completion)
                    # Display the existing ratings
                    #ratings = [(doc.to_dict()["rating"], doc.to_dict()["user"]) for doc in ratings_ref.stream() if doc.to_dict()["completion_id"] == completion_id]
                    ratings = []
                    query = ratings_ref.where("completion_id", "==", completion_id)
                    for doc in query.stream():
                        rating = doc.to_dict()["rating"]
                        user = doc.to_dict()["user"]
                        ratings.append((rating, user))
                    st.write(f"{len(ratings)} ratings")
                    if display_ratings:
                    # Get the ratings for this completion
                        for rating, user in ratings:
                            st.write(f"Rating: {rating} by {user}")
                        # mean rating
                        st.write(f"Mean rating: {np.mean([rating for rating, user in ratings])}")
                    # create a form
                    with st.form(f"rating_{i}"):
                        rating = st.slider("Rate this completion", min_value=0, max_value=5, key=completion)
                        st.write(f"Rating: {rating}")
                        submit_button = st.form_submit_button("Submit")
                        if submit_button:
                            st.balloons()
                            #st.write(f"Adding a rating of {rating} for completion {completion}")
                            # Add the rating to the database
                            ratings_ref.add({"prompt_id": prompt_id, "completion_id":completion_id, "completion": completion, "rating": rating, "user": user, "model": model,
                                            "date": SERVER_TIMESTAMP})
                # grey line
                st.markdown("---")


