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

openai.api_key = st.secrets["API_KEY"]


# Who are you?
user = st.sidebar.text_input("Who are you?", value="anonymous")

model = st.sidebar.selectbox("Select a model", options=["text-davinci-003", "gpt-3.5-turbo", "code-davinci-002"])

# Add a selectbox to select a prompt
# get brands from prompts_ref
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
#prompts = [doc.to_dict()["prompt"] for doc in prompts_ref.stream() if doc.to_dict()["brand"] == brand and datetime.datetime.strptime(doc.to_dict()["date"], "%d/%m/%Y %H:%M:%S").date() >= min_date]
@st.cache_resource()
def load_prompts(brand, min_date):
    print("Loading prompts..")
    with st.spinner("Loading prompts.."):
        prompts = []
        prompt_ids = []
        query = prompts_ref.where("brand", "==", brand).where("date", ">=", datetime.datetime.combine(min_date, datetime.time.min))
        for doc in query.stream():
            prompt = doc.to_dict()["prompt"]
            prompt_id = doc.id
            prompts.append(prompt)
            prompt_ids.append(prompt_id)
    print("Done!")
    return prompts, prompt_ids

@st.cache_resource()
def load_completions(prompt_ids, model):
    all_completions_dic = {}
    with st.spinner("Loading completions.."):
        for prompt_id in prompt_ids:
            query = completions_ref.where("prompt_id", "==", prompt_id).where("model", "==", model)
            for doc in query.stream():
                completion_dic = doc.to_dict()
                all_completions_dic[doc.id] = completion_dic
    with st.spinner("Loading ratings.."):
        for completion_id in all_completions_dic:
            if "ratings" not in all_completions_dic[completion_id]:
                all_completions_dic[completion_id]["ratings"] = []
            if "date" not in all_completions_dic[completion_id]:
                all_completions_dic[completion_id]["date"] = "Unknown"
            if "truncated" not in all_completions_dic[completion_id]:
                all_completions_dic[completion_id]["truncated"] = "Unknown"
        # find all ratings for all completions
        #FIXME: this is faster when there are few ratings, but maybe the previous way is better when there are many ratings
        # Slice in bin of 10 to work with firestore query limit
        for i in range(0, len(all_completions_dic), 10):
            query = ratings_ref.where("completion_id", "in", list(all_completions_dic.keys())[i:i+10])
            for doc in query.stream():
                rating = doc.to_dict()["rating"]
                user = doc.to_dict()["user"]
                completion_id = doc.to_dict()["completion_id"]
                all_completions_dic[completion_id]["ratings"].append((rating, user))

    return all_completions_dic

# @st.cache_resource()
# def get_attribute_per_completion(completions_id):
#     #TODO get completions id and attribute at the same time
#     attribute_per_completion = {}
#     for completion_id in completions_id:
#         if "ratings" not in attribute_per_completion:
#             attribute_per_completion["ratings"] = {}
#         if "date" not in attribute_per_completion:
#             attribute_per_completion["date"] = {}
#         if "truncated" not in attribute_per_completion:
#             attribute_per_completion["truncated"] = {}
#         if "completion" not in attribute_per_completion:
#             attribute_per_completion["completion"] = {}
#         query = ratings_ref.where("completion_id", "==", completion_id)
#         ratings = []
#         for doc in query.stream():
#             rating = doc.to_dict()["rating"]
#             user = doc.to_dict()["user"]
#             ratings.append((rating, user))
#         completion_doc = completions_ref.document(completion_id).get()
#         if "date" not in completion_doc.to_dict():
#             date = "Unknown"
#         else:
#             date = completion_doc.to_dict()["date"]
#         if "truncated" in completion_doc.to_dict():
#             truncated = completion_doc.to_dict()["truncated"]
#         else:
#             truncated = "Unknown"
#         completion = completion_doc.to_dict()["completion"]
#         attribute_per_completion["ratings"][completion_id] = ratings
#         attribute_per_completion["date"][completion_id] = date
#         attribute_per_completion["truncated"][completion_id] = truncated
#         attribute_per_completion["completion"][completion_id] = completion

#     return attribute_per_completion

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
            prompts_ref.add({"brand": brand, "prompt": new_prompt, "user": user,
                             "hash": hash(new_prompt),
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
        #st.write(f"Prompt created by {user} on {}") #TODO
        # get current prompt id
        prompt_id_for_prompt = [prompt_ids[i] for i, prompt_ in enumerate(prompts) if prompt_ == prompt]
        # There may be several prompts with the same text
        # But maybe we should not allow that
        st.write(f"Prompt ids: {prompt_id_for_prompt}")
        #completions = [doc.to_dict()["completion"] for doc in completions_ref.stream() if doc.to_dict()["prompt_id"] == prompt_id]
        completions_dic = load_completions(prompt_id_for_prompt, model)
        # show number of completions and ratings
        st.header(f"{len(completions_dic)} completions with {sum([len(completions_dic[completion_id]['ratings']) for completion_id in completions_dic])} ratings")
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
                    #TODO how to choose the prompt id?
                    completions_ref.add({"prompt_id": prompt_ids[0], "hash": hash(prompt), "completion": completion, "truncated": truncated, "user": user, "model": model, 
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
        if sort_by == "Rating":
            # Get the completions and sort them by rating
            # get the mean rating for each completion
            # sort the completions by mean rating
            mean_ratings = np.array([np.mean([e[0] for e in completions_dic[key]["ratings"]]) for key in completions_dic.keys()])
            indices = np.argsort(mean_ratings)[::-1]
            print(f"Mean ratings: {mean_ratings}")
            print(f"Indices: {indices}")
            nan_idx = np.isnan(mean_ratings[indices])
            sort_idx = np.concatenate((indices[~nan_idx], indices[nan_idx]))
            #mean_ratings = [np.mean([e[0] for e in completions_dic["ratings"][completion_id]]) for completion_id in completions_dic.keys()]
            completions_ids = [list(completions_dic.keys())[i] for i in sort_idx]

        elif sort_by == "Date":
            # Sort the completions by date
            completions_ids = sorted(completions_dic.keys(), key=lambda completion_id: completions_dic[completion_id]["date"], reverse=True)
        elif sort_by == "Random":
            # Get the completions and sort them randomly
            completions_ids = sorted(completions_dic.keys(), key=lambda completion_id: random.random(), reverse=True)
        # Loop over the completions
        for i, completion_id in enumerate(completions_ids):
            # Get the completion
            #completion = [doc.to_dict()["completion"] for doc in completions_ref.stream() if doc.id == completion_id][0]
            # query the completions_ref to get the completion
            ratings = completions_dic[completion_id]["ratings"]
            date = completions_dic[completion_id]["date"]
            truncated = completions_dic[completion_id]["truncated"]
            completion = completions_dic[completion_id]["completion"]
            # Write the number of completions
            # create a box to display the completion
            if not hide_truncated or truncated == "False":
                with st.expander(f"Completion {i}, created: {date}, truncated: {truncated}", expanded=True):
                    # Write the completion
                    st.write(completion)
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
                            ratings_ref.add({"prompt_id": prompt_ids[0], "completion_id":completion_id, "hash":hash(completion), "rating": rating, "user": user, "model": model,
                                            "date": SERVER_TIMESTAMP})
                # grey line
                st.markdown("---")


st.sidebar.header("More info")
st.sidebar.subheader("Ratings guidelines")
st.sidebar.write("0: The style is not here at all")
st.sidebar.write("1: You can barely see the style")
st.sidebar.write("2: You can see the style, but it is not very good")
st.sidebar.write("3: The style is pretty much here, but there are some mistakes")
st.sidebar.write("4: Something is kinda weird, but a client would be happy with this")
st.sidebar.write("5: A professional writer would write something like this")
