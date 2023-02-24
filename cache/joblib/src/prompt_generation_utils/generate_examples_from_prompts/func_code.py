# first line: 6
@memory.cache
def generate_examples_from_prompts(prompt, 
                                   n_examples_per_prompt=10, 
                                   engine="text-davinci-003", 
                                   max_tokens=250, 
                                   temperature=0.7,
                                   top_p=1,
                                   frequency_penalty=0,
                                   presence_penalty=0,
                                   stop=["END"]):
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
    return [choice["text"] for choice in response["choices"]]
