from tqdm import tqdm
import openai
import pandas as pd
import numpy as np
import sklearn
import numpy as np
 # Train a classifier to distinguish between the true and generated examples
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Memory
memory = Memory("/Users/leo/VSCProjects/newletters/cache", verbose=0)




@memory.cache
def generate_examples_from_prompts(prompt_prefix, prompt_suffix, prompt_list, n_examples_per_prompt=10, 
                                   engine="text-davinci-003", 
                                   max_tokens=250, 
                                   temperature=0.7,
                                   top_p=1,
                                   frequency_penalty=0,
                                   presence_penalty=0):
    """
    Generate examples from a list of prompts, with openai API
    """
    examples = []
    for prompt in prompt_list:
        prompt = prompt_prefix + prompt + prompt_suffix
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=n_examples_per_prompt,
            stop=["END"]
        )
        examples.extend([choice["text"] for choice in response["choices"]])

    # Add the stop token to the end of the examples
    examples = [example + "\nEND" for example in examples]
    return examples


@memory.cache
def get_embedding(examples, model="text-embedding-ada-002"):
   #text = text.replace("\n", " ")
   embeddings =  openai.Embedding.create(input = examples, model=model)['data']
   # Create a numpy array of the embeddings
   embeddings = np.array([np.array(embedding["embedding"]) for embedding in embeddings])
   return embeddings
 

@memory.cache
def score_examples_classifier(true_examples, generated_examples, n_true=None, n_trials=1, show_examples=False):
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
        res_dic[clf_name]["mean_generated_proba"] = np.zeros(n_trials)
        res_dic[clf_name]["mean_true_proba"] = np.zeros(n_trials)

        

    if n_true is None:
        n_true = len(embeddings_true)
    for i in range(n_trials):
        # fix seed for reproducibility
        np.random.seed(i)
        chosen_true_examples_indices = np.random.choice(len(embeddings_true), n_true, replace=False)
        embeddings_true_chosen = embeddings_true[chosen_true_examples_indices]

        X = np.concatenate([embeddings_generated, embeddings_true_chosen])
        y = np.concatenate([np.zeros(len(embeddings_generated)), np.ones(n_true)])


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)


        for clf, clf_name in zip(clf_list, clf_names):
            clf = clf(random_state=i)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            res_dic[clf_name]["accuracy"][i] = accuracy_score(y_test, y_pred)
            res_dic[clf_name]["precision"][i] = precision_score(y_test, y_pred)
            res_dic[clf_name]["recall"][i] = recall_score(y_test, y_pred)
            res_dic[clf_name]["f1"][i] = f1_score(y_test, y_pred)
            res_dic[clf_name]["min_accuracy"][i] = max(1 - np.mean(y_test), np.mean(y_test))
            y_pred_proba = clf.predict_proba(X_test)
            # compute the mean probablity of the generated examples being true
            res_dic[clf_name]["mean_generated_proba"][i] = np.mean(y_pred_proba[y_test == 0, 1])
            # compute the mean probablity of the true examples being true
            res_dic[clf_name]["mean_true_proba"][i] = np.mean(y_pred_proba[y_test == 1, 1])


            if show_examples:
                # Sort the generated examples by their probability of being true
                y_pred_proba = clf.predict_proba(X_test)
                y_pred_proba_true = y_pred_proba[:, 1]
                y_pred_proba_true_sorted_indices = np.argsort(y_pred_proba_true)[::-1]
                print(f"--------- Classifier: {clf_name} ---------")
                print("Generated examples sorted by probability of being true:")
                print("----------------------------------------")
                for j in y_pred_proba_true_sorted_indices:
                    if y_test[j] == 0: # generated example
                        print(f"Example: {generated_examples[j]}")
                        print(f"Probability of being true: {y_pred_proba_true[j]}")
                        print(f"True label: {y_test[j]}")
                        print(f"Predicted label: {y_pred[j]}")
                        print("")
                        print("----------------------------------------")

                # Sort the true examples by their probability of being false
                y_pred_proba_false = y_pred_proba[:, 0]
                y_pred_proba_false_sorted_indices = np.argsort(y_pred_proba_false)[::-1]
                print("True examples sorted by probability of being false:")
                print("----------------------------------------")
                for j in y_pred_proba_false_sorted_indices:
                    if y_test[j] == 1:
                        print(f"Example: {true_examples[j]}")
                        print(f"Probability of being true: {y_pred_proba_true[j]}")
                        print(f"True label: {y_test[j]}")
                        print(f"Predicted label: {y_pred[j]}")
                        print("")
                        print("----------------------------------------")



    
    return res_dic




