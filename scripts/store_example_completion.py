import openai
import sklearn
import pandas as pd
import functools
from src.prompt_scoring_utils import get_embedding, score_examples_classifier
from src.prompt_generation_utils import generate_examples_from_prompts_cached
import pickle
import numpy as np
from transformers import GPT2Tokenizer
import time
from google.cloud import firestore
import datetime
import streamlit as st
import json
from google.oauth2 import service_account




if __name__ == "__main__":

    user = "Leo"

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    # Get OPENAI API_KEY from streamlit secret
    openai.api_key = st.secrets["API_KEY"]

    
    def get_firestore_client():
        key_dict = json.loads(st.secrets["textkey"])
        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds)
        return db

    # Get the prompts from Firestore
    db = get_firestore_client()

    db = get_firestore_client()
    prompts_ref = db.collection("prompts")
    completions_ref = db.collection("completions")
    print("firestore database loaded")
    
    # Load the data
    data = pd.read_json("data/frichti_clean_prepared (2).jsonl", lines=True)
    print("Samples loaded")

    brand = "Frichti"
    #model = "gpt-3.5-turbo"
    model = "code-davinci-002"

    true_examples = data["completion"].values.tolist()

    dic_list = []

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    max_tokens = 300

    file_name = "results/prompt_scoring_results_best_chat.pkl"


    n_seeds = 20
    n_examples_per_prompt = 4
    all_seeds = list(range(n_seeds))

    #all_seeds = [19, 8, 4, 11, 17, 1, 14, 0, 18, 7]
    for seed in all_seeds:
        seed = seed
        prompts = sklearn.utils.resample(true_examples, n_samples=4, random_state=seed)
        prompt_prefix = "Voici des exemples de newsletter Frichti:\n"
        for i, prompt in enumerate(prompts):
            prompt_prefix += f"Example {i + 1} \n"
            prompt_prefix += prompt
            prompt_prefix += "\n"

        #print("Prompt prefix: ", prompt_prefix)
        # Generate examples from the prompt prefix
        #prompt_suffix = "Ecris une courte newsletter Frichti du même style que les exemples:"
        prompt_suffix = f"Example {len(prompts) + 1} \n" #for model not instruction tuned
        complete_prompt = prompt_prefix  + prompt_suffix
        #add to the database and get prompt id
        # Check if it's already in the database with the hash
        #prompts_ref = db.collection("prompts")
        #prompts = prompts_ref.where("hash", "==", hash(complete_prompt)).stream()
        prompt_ref = prompts_ref.document()
        prompt_ref.set({
            "brand": brand,
            "prompt": complete_prompt,
            "sha": sha,
            "user": user,
            "date": datetime.datetime.now(),
            "hash": hash(complete_prompt),
        })
        prompt_id = prompt_ref.id

        print("Generating completions")
        try:
            response = generate_examples_from_prompts_cached(complete_prompt, n_examples_per_prompt=10, max_tokens=max_tokens,
                                                                engine=model, check_if_truncated=True)
        except openai.error.RateLimitError as e:
            print(e)
            print("Sleeping for 1 minute")
            time.sleep(60)
            response = generate_examples_from_prompts_cached(complete_prompt, n_examples_per_prompt=10, max_tokens=max_tokens,
                                                                engine=model, check_if_truncated=True)
            # generated_examples.extend(generate_examples_from_prompts(prompt_prefix, prompt_suffix, prompts, n_examples_per_prompt=80, max_tokens=max_tokens,
            #                                                     engine="text-davinci-003"))
        print("Generation completed !")
        generated_examples, truncated_list = zip(*response)
        print(f"Generated {len(generated_examples)} examples")
        print(f"Number of truncated examples: {sum(truncated_list)}")

        # Store the completions in the database
        for i, completion in enumerate(generated_examples):
            completion_ref = completions_ref.document()
            completion_ref.set({
                "brand": brand,
                "prompt_id": prompt_id,
                "completion": completion,
                "sha": sha,
                "user": user,
                "date": datetime.datetime.now(),
                "prompt": complete_prompt,
                "truncated": truncated_list[i],
                "model": model,
            })
        print("Completions stored in the database")
