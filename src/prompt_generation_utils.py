import openai
import transformers
from joblib import Memory

memory = Memory("../cache", verbose=0)


# @memory.cache
def generate_examples_from_prompts(prompt, 
                                   n_examples_per_prompt=10, 
                                   engine="text-davinci-003", 
                                   max_tokens=250, 
                                   temperature=0.7,
                                   top_p=1,
                                   frequency_penalty=0,
                                   presence_penalty=0,
                                   stop=["END"],
                                   check_if_truncated=False): # for caching)
    """
    Generate examples from a list of prompts, with openai API
    """
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=n_examples_per_prompt,
        stop=stop,
    )
    if check_if_truncated:
        # check if the number of tokens is equal to max_tokens
        # if yes, then the example is truncated
        # tokenize
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        for choice in response["choices"]:
            tokens = tokenizer.encode(choice["text"])
            if len(tokens) == max_tokens:
                choice["truncated"] = True
            else:
                choice["truncated"] = False
        return [(choice["text"], choice["truncated"]) for choice in response["choices"]]
    
    return [choice["text"] for choice in response["choices"]]

@memory.cache
def generate_examples_from_prompts_cached(prompt, 
                                   n_examples_per_prompt=10, 
                                   engine="text-davinci-003", 
                                   max_tokens=250, 
                                   temperature=0.7,
                                   top_p=1,
                                   frequency_penalty=0,
                                   presence_penalty=0,
                                   stop=["END"],
                                   check_if_truncated=False):
    return generate_examples_from_prompts(prompt,
                                   n_examples_per_prompt=n_examples_per_prompt,
                                   engine=engine,
                                   max_tokens=max_tokens,
                                   temperature=temperature,
                                   top_p=top_p,
                                   frequency_penalty=frequency_penalty,
                                   presence_penalty=presence_penalty,
                                   stop=stop,
                                   check_if_truncated=check_if_truncated)
    
