import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def query_with_cag(knowledge_base, query, model="gpt-4o", temperature=0.2, max_tokens=500):
    # Conceptually, this resembles Cache-Augmented Generation (CAG)
    # because we are providing the entire knowledge base as context in the prompt.
    # However, the actual caching mechanism (KV Caching) is handled *internally* by the OpenAI API.
    # We are not explicitly pre-computing or managing the KV cache like in the transformer-based example.
    # OpenAI's infrastructure likely caches parts of the prompt (including the knowledge base)
    # to speed up processing if the same base context is used repeatedly,
    # but this is an optimization on their end, not direct user-controlled CAG.
    # The effectiveness depends on the API's internal caching strategy and whether the context fits.

    system_message = "You are an AI assistant with expert knowledge. Answer questions based only on the provided context."
    prompt = f"Context (this is your knowledge base, use this information to answer the question):\n{knowledge_base}\n\nQuery: {query}"

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content.strip()

def main():
    knowledge_base_path = "none-technical.md"
    
    knowledge_base = load_knowledge_base(knowledge_base_path)
    if not knowledge_base:
        return
    
    query = "What is Cache-Augmented Generation (CAG)?"

    print("\nGenerating response...\n")
    response = query_with_cag(knowledge_base, query)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
