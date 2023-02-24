import openai
import sklearn
import pandas as pd
import functools
api_key = ""
openai.api_key = api_key
# Load the data
data = pd.read_json("frichti_clean_prepared (3).jsonl", lines=True)

true_examples = data["completion"].values.tolist()
@functools.lru_cache(maxsize=1000, typed=False)
def generate_examples_from_prompts(prompt_prefix, prompt_suffix, prompt_list, n_examples_per_prompt=10):
    """
    Generate examples from a list of prompts, with openai API
    """
    examples = []
    for prompt in prompt_list:
        prompt = prompt_prefix + prompt + prompt_suffix
        for i in range(n_examples_per_prompt):
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=250,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["END"]
            )
            examples.append(response["choices"][0]["text"])
    return examples

# Take 4 examples at random for the prompt prefix, and remove them from the dataset
# The examples should be long enough to be representative of the dataset

seed = 42
prompts = sklearn.utils.resample(true_examples, n_samples=4, random_state=seed)

prompt_prefix = "Voici des exemples de newsletter Frichti:\n"
for i, prompt in enumerate(prompts):
    prompt_prefix += f"Example {i + 1} \n"
    prompt_prefix += prompt
    prompt_prefix += "\n"

print("Prompt prefix: ", prompt_prefix)
# Generate examples from the prompt prefix
prompt_suffix = "Ecris une nouvelle newsletter Frichti du même style que les exemples:"
prompts = ('',) # tuple to be able to use lru_cache
generated_examples = generate_examples_from_prompts(prompt_prefix, prompt_suffix, prompts, n_examples_per_prompt=3)
for i, example in enumerate(generated_examples):
    print(f"Example {i}: {example}")

# Embed the true and generated examples with the openai api and train a classifier
import numpy as np

@functools.lru_cache(maxsize=1000, typed=False)
def get_embedding(examples, model="text-embedding-ada-002"):
   #text = text.replace("\n", " ")
   embeddings =  openai.Embedding.create(input = examples, model=model)['data']
   # Create a numpy array of the embeddings
   embeddings = np.array([np.array(embedding["embedding"]) for embedding in embeddings])
   return embeddings
 


# Train a classifier to distinguish between the true and generated examples
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@functools.lru_cache(maxsize=1000, typed=False)
def score_examples(true_examples, generated_examples, n_true=None, n_trials=1):
    embeddings_generated = get_embedding(tuple((generated_examples)))
    embeddings_true = get_embedding(tuple(true_examples))

    res_dic  = {}
    clf_list = [LogisticRegression, RandomForestClassifier]
    clf_names = ["LogisticRegression", "RandomForestClassifier"]
    for clf, clf_name in zip(clf_list, clf_names):
        # Create empty arrays for each classifier and each metric
        res_dic[clf_name] = {}
        res_dic[clf_name]["accuracy"] = np.zeros(n_trials)
        res_dic[clf_name]["precision"] = np.zeros(n_trials)
        res_dic[clf_name]["recall"] = np.zeros(n_trials)
        res_dic[clf_name]["f1"] = np.zeros(n_trials)
        res_dic[clf_name]["min_accuracy"] = np.zeros(n_trials)
        

    if n_true is None:
        n_true = len(embeddings_true)
    for i in range(n_trials):
        chosen_true_examples_indices = np.random.choice(len(embeddings_true), n_true, replace=False)
        embeddings_true_chosen = embeddings_true[chosen_true_examples_indices]

        X = np.concatenate([embeddings_generated, embeddings_true[:n_true]])
        y = np.concatenate([np.zeros(len(embeddings_generated)), np.ones(n_true)])


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        for clf, clf_name in zip(clf_list, clf_names):
            clf = clf(random_state=i)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            res_dic[clf_name]["accuracy"][i] = accuracy_score(y_test, y_pred)
            res_dic[clf_name]["precision"][i] = precision_score(y_test, y_pred)
            res_dic[clf_name]["recall"][i] = recall_score(y_test, y_pred)
            res_dic[clf_name]["f1"][i] = f1_score(y_test, y_pred)
            res_dic[clf_name]["min_accuracy"][i] = max(1 - np.mean(y_test), np.mean(y_test))
    
    return res_dic


