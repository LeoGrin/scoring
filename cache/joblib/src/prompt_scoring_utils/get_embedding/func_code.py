# first line: 44
@memory.cache
def get_embedding(examples, model="text-embedding-ada-002"):
   #text = text.replace("\n", " ")
   embeddings =  openai.Embedding.create(input = examples, model=model)['data']
   # Create a numpy array of the embeddings
   embeddings = np.array([np.array(embedding["embedding"]) for embedding in embeddings])
   return embeddings
