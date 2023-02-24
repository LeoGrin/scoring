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



if __name__ == "__main__":

    user = "Leo"

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    api_key = ""
    openai.api_key = api_key

    
    def get_firestore_client():
        db = firestore.Client.from_service_account_json("firestore-key.json")
        return db

    db = get_firestore_client()
    prompts_ref = db.collection("prompts")
    completions_ref = db.collection("completions")
    print("firestore database loaded")
    
    # Load the data
    data = pd.read_json("data/frichti_clean_prepared (2).jsonl", lines=True)
    print("Samples loaded")

    brand = "Frichti"

    true_examples = data["completion"].values.tolist()

    dic_list = []

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    max_tokens = 300

    file_name = "results/prompt_scoring_results_best.pkl"


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
        prompt_suffix = "Ecris une courte newsletter Frichti du même style que les exemples:"
        complete_prompt = prompt_prefix  + prompt_suffix
        # add to the database and get prompt id
        prompt_ref = prompts_ref.document()
        prompt_ref.set({
            "brand": brand,
            "prompt": complete_prompt,
            "sha": sha,
            "user": user,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        prompt_id = prompt_ref.id

        print("Generated completions")
        try:
            response = generate_examples_from_prompts_cached(complete_prompt, n_examples_per_prompt=1, max_tokens=max_tokens,
                                                                engine="text-davinci-003", check_if_truncated=True)
        except openai.error.RateLimitError as e:
            print(e)
            print("Sleeping for 1 minute")
            time.sleep(60)
            response = generate_examples_from_prompts_cached(complete_prompt, n_examples_per_prompt=1, max_tokens=max_tokens,
                                                                engine="text-davinci-003", check_if_truncated=True)
            # generated_examples.extend(generate_examples_from_prompts(prompt_prefix, prompt_suffix, prompts, n_examples_per_prompt=80, max_tokens=max_tokens,
            #                                                     engine="text-davinci-003"))
        print("Generation completed !")
        generated_examples, truncated_list = zip(*response)
        print(f"Generated {len(generated_examples)} examples")
        n_tokens = [len(tokenizer.encode(example)) for example in generated_examples]
        print("Number of tokens in generated examples: ", n_tokens)

        # Remove the truncated examples
        print("Number of examples before removing truncated examples: ", len(generated_examples))
        
        generated_examples = [example for example, n_tokens in zip(generated_examples, n_tokens) if n_tokens < max_tokens - 2]
        print("Number of examples after removing truncated examples: ", len(generated_examples))
        if len(generated_examples) == 0:
            print("No examples generated, skipping this seed")
            continue
        # Score the examples
        print("Scoring examples...")
        res_dic = score_examples_classifier(tuple(true_examples), tuple(generated_examples), n_true=len(generated_examples), n_trials=10, show_examples=False)
        print("Scoring done !")
        res_dic["seed"] = seed
        dic_list.append(res_dic)
        # Save the results
        with open(file_name, "wb") as f:
            pickle.dump(dic_list, f)

    # Print the best prompt according to logreg accuracy
    log_reg_acc = [np.mean(dic["LogisticRegression"]["accuracy"]) for dic in dic_list]
    # print prompts for the 10 best seeds
    best_seeds_indices = np.argsort(log_reg_acc)[:10]
    print("----------------------------------")
    print("Best seeds according to LogisticRegression accuracy: ", [dic_list[seed_indice]["seed"] for seed_indice in best_seeds_indices])
    for seed_indice in best_seeds_indices:
        seed = dic_list[seed_indice]["seed"]
        print(f"Seed {seed}: {dic_list[seed_indice]['seed']}")
        print(f"LogisticRegression accuracy: {np.mean(dic_list[seed_indice]['LogisticRegression']['accuracy'])}")
        print(f"(Min accuracy:: {np.mean(dic_list[seed_indice]['LogisticRegression']['min_accuracy'])})")
        print(f"LogisticRegression f1: {np.mean(dic_list[seed_indice]['LogisticRegression']['f1'])}")
        print(f"LogisticRegression precision: {np.mean(dic_list[seed_indice]['LogisticRegression']['precision'])}")
        print(f"LogisticRegression recall: {np.mean(dic_list[seed_indice]['LogisticRegression']['recall'])}")
        print(f"LogisticRegression mean_true_proba - mean_generated_proba: {np.mean(dic_list[seed_indice]['LogisticRegression']['mean_true_proba'] - dic_list[seed_indice]['LogisticRegression']['mean_generated_proba'])}")
    
    # Same for Random Forest
    rf_acc = [np.mean(dic["RandomForestClassifier"]["accuracy"]) for dic in dic_list]
    # print prompts for the 10 best seeds
    best_seeds_indices = np.argsort(rf_acc)[:10]
    print("----------------------------------")
    print("Best seeds according to RandomForestClassifier accuracy: ", [dic_list[seed_indice]["seed"] for seed_indice in best_seeds_indices])
    for seed_indice in best_seeds_indices:
        seed = dic_list[seed_indice]["seed"]
        print(f"Seed {seed}: {dic_list[seed_indice]['seed']}")
        print(seed_indice)
        print(f"RandomForestClassifier accuracy: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['accuracy'])}")
        print(f"(Min accuracy: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['min_accuracy'])})")
        print(f"RandomForestClassifier f1: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['f1'])}")
        print(f"RandomForestClassifier precision: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['precision'])}")
        print(f"RandomForestClassifier recall: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['recall'])}")
        print(f"RandomForestClassifier mean_true_proba - mean_generated_proba: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['mean_true_proba'] - dic_list[seed_indice]['RandomForestClassifier']['mean_generated_proba'])}")
    
    # Now order on the difference between the mean true and generated proba
    diff_proba = [np.mean(dic["LogisticRegression"]["mean_true_proba"] - dic["LogisticRegression"]["mean_generated_proba"]) for dic in dic_list]
    best_seeds_indices = np.argsort(diff_proba)[:10]
    print("----------------------------------")
    print("Best seeds according to LogisticRegression mean_true_proba - mean_generated_proba: ", [dic_list[seed_indice]["seed"] for seed_indice in best_seeds_indices])
    for seed_indice in best_seeds_indices:
        seed = dic_list[seed_indice]["seed"]
        print(f"Seed {seed}: {dic_list[seed_indice]['seed']}")
        print(f"LogisticRegression accuracy: {np.mean(dic_list[seed_indice]['LogisticRegression']['accuracy'])}")
        print(f"LogisticRegression f1: {np.mean(dic_list[seed_indice]['LogisticRegression']['f1'])}")
        print(f"LogisticRegression precision: {np.mean(dic_list[seed_indice]['LogisticRegression']['precision'])}")
        print(f"LogisticRegression recall: {np.mean(dic_list[seed_indice]['LogisticRegression']['recall'])}")
        print(f"LogisticRegression mean_true_proba - mean_generated_proba: {np.mean(dic_list[seed_indice]['LogisticRegression']['mean_true_proba'] - dic_list[seed_indice]['LogisticRegression']['mean_generated_proba'])}")

    # Same for Random Forest
    diff_proba = [np.mean(dic["RandomForestClassifier"]["mean_true_proba"] - dic["RandomForestClassifier"]["mean_generated_proba"]) for dic in dic_list]
    best_seeds_indices = np.argsort(diff_proba)[:10]
    print("----------------------------------")
    print("Best seeds according to RandomForestClassifier mean_true_proba - mean_generated_proba: ", [dic_list[seed_indice]["seed"] for seed_indice in best_seeds_indices])
    for seed_indice in best_seeds_indices:
        seed = dic_list[seed_indice]["seed"]
        print(f"Seed {seed}: {dic_list[seed_indice]['seed']}")
        print(f"RandomForestClassifier accuracy: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['accuracy'])}")
        print(f"RandomForestClassifier f1: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['f1'])}")
        print(f"RandomForestClassifier precision: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['precision'])}")
        print(f"RandomForestClassifier recall: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['recall'])}")
        print(f"RandomForestClassifier mean_true_proba - mean_generated_proba: {np.mean(dic_list[seed_indice]['RandomForestClassifier']['mean_true_proba'] - dic_list[seed_indice]['RandomForestClassifier']['mean_generated_proba'])}")
