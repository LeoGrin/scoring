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
    print("Generating examples from prompt...")
    if engine == "gpt-3.5-turbo":
        # the gpt-3.5-turbo model is a bit different from the others
        messages=[         
        {"role": "system", "content": "You are a writer."},         
        {"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(   
            model="gpt-3.5-turbo",   
            messages=messages,
            n=n_examples_per_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        replies = [response["choices"][i]["message"]["content"] for i in range(n_examples_per_prompt)]
    else:
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
        replies = [response["choices"][i]["text"] for i in range(n_examples_per_prompt)]
    if check_if_truncated:
        # check if the number of tokens is equal to max_tokens
        # if yes, then the example is truncated
        # tokenize
        truncated = []
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        for reply in replies:
            print(reply)
            tokens = tokenizer.encode(reply)
            if len(tokens) == max_tokens:
                truncated.append(True)
            else:
                truncated.append(False)
        return [(reply, truncated[i]) for i, reply in enumerate(replies)]
    
    return replies

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
    
