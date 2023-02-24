# first line: 18
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
