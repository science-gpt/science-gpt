from langchain_openai import AzureOpenAIEmbeddings

# Azure OpenAI configuration
endpoint = "https://ai-useastsciencegptaihub287254834673.openai.azure.com"
api_key = "7eb57e71ebb24b9c8631733c2b353d89"
model_name = "text-embedding-3-large"

# Initialize the Azure OpenAI Embedding model
embedding_model = AzureOpenAIEmbeddings(
    model=model_name,
    azure_endpoint=endpoint,
    api_key=api_key,  # If not provided, reads from AZURE_OPENAI_API_KEY environment variable
)

# Test a basic query to check the embedding response
try:
    response = embedding_model.embed_query("This is a test")
    print("Embedding:", response)
except Exception as e:
    print(f"Error: {e}")
