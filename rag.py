import os
from pathlib import Path
import hashlib
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(separators=['.\n'],chunk_size=1500,chunk_overlap=50)

query = "What is Cache-Augmented Generation (CAG)?"
top_k = 2
file_path = 'none-technical.md'

Path("embeddings").mkdir(exist_ok=True)
file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
embedding_file = Path("embeddings") / f"{file_hash}.faiss"

if embedding_file.exists():
    vector_db = FAISS.load_local(str(embedding_file), embeddings, allow_dangerous_deserialization=True)
else:
    text = Path(file_path).read_text(encoding='utf-8')
    documents = text_splitter.create_documents([text])
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(str(embedding_file))

result = vector_db.similarity_search(query, k=top_k)

contents = [doc.page_content for doc in result]



from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
prompt = f"answer the query {query} base on following contents:\n {contents}"
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

response = r.choices[0].message.content
print(response)
