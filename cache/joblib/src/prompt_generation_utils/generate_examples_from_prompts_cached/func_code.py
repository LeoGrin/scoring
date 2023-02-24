# first line: 48
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
