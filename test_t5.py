import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


# Load FAISS index and perform similarity search
# Increase `k` to retrieve more relevant documents
def load_faiss_and_search(query, index_path="DataIndex", k=5):  # Adjust k value here
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    docs = faiss_index.similarity_search(query, k=k)
    return docs

# Generate an answer using Flan-T5 model
def generate_answer_with_flan_t5(context, question):
    model_name = "google/flan-t5-large"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Combine the context and the question into a prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    # Tokenize input and ensure max_length does not exceed the model's token limit
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode the output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# Main function to handle user query and generate response
def user_query(query):
    # Load FAISS index and search for relevant documents
    docs = load_faiss_and_search(query)

    # Combine the text of the retrieved documents to use as context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Limit the context length to avoid exceeding the model's max token limit (512 tokens)
    max_context_length = 512  # Adjust as needed
    context = context[:max_context_length]  # Truncate the context

    # Generate an answer using Flan-T5
    answer = generate_answer_with_flan_t5(context, query)
    return answer


# Example usage
if __name__ == "__main__":
    while True:
        query = input("Please enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = user_query(query)
        print(f"Answer:\n{answer}")
