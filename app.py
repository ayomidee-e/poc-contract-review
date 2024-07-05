import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import openai
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma


# Function to check if uploaded document is a contract
def check_if_contract(text):
    contract_keywords = ["agreement", "party", "payment", "terms",
                         "conditions", "confidentiality", "termination",
                         "obligations"]
    return any(keyword.lower() in text.lower() for keyword in 
               contract_keywords)


# Streamlit app
def main():
    st.title('Contract Review Assistant')

    # Load OpenAI API key from .env file
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    # Define predefined questions
    predefined_questions = [
        "What type of Contract is this?",
        "Who are the Parties involved?, Start with Parties Invol"
        "What are the payment terms?, Start with: 'Payment Terms:', If there \
            are no payment terms in the contract uploaded, state that it \
                wasn't mentioned",
        "What is the duration of the contract?, Start with: \
            'Contract Duration', If the duration of the contract wasn't \
                mentioned in the uploaded contract, state that it wasn't \
                    mentioned",
        "What are the termination conditions?, Start with: \
            'Termination Condition', If the Termination condition isn't \
                mentioned in the uploaded contract, state that it wasn't \
                    mentioned",
        "Are there any confidentiality clauses?, Start with: \
            'Confidentiality Clauses', If there are no confidentiality clause \
                mentioned in the contract uploaded, state that it wasn't \
                    mentioned",
        "What are the obligations of each party?, Start with: \
            'Obligation of each party', If there are no obligation of any \
                party mentioned in the uploaded contract, state that it \
                    wasn't mentioned"
    ]

    # Get source document input
    user_doc = st.file_uploader("Upload Your Contract", type="pdf")

    # Check if the 'Review Contract' button is clicked
    if st.button("Review Contract"):
        # Validate input
        if not user_doc:
            st.write("Please upload your contract.")
        else:
            try:
                # Save uploaded file temporarily to disk
                # load and split the file into pages, delete temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(user_doc.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load_and_split()
                os.remove(tmp_file.name)

                # Check if the document appears to be a contract
                document_text = "\n".join(page.page_content for page in pages)
                if not check_if_contract(document_text):
                    st.write("The uploaded document does not appear to be a \
                             contract.")
                    return

                # Define the template for the language model prompt
                template = """You are a language model AI developed for \
                    summarizing contract documents.\
                    You are a friendly virtual assistant designed to provide \
                        detailed summaries based on uploaded contracts.\
                    Your objective is to provide accurate and relevant \
                        summaries of the contract strictly based on the \
                            documents uploaded.\
                    In your responses, ensure a tone of friendliness but \
                        professionalism.\

                    Here are some specific interaction scenarios to guide \
                        your responses:
                    - Always start only your first summary with: "Hello, \
                        here is a summary of your contract."
                    - Only answer from what is in the uploaded contract, \
                        focusing on the important details as stated.

                    Contract Content: {context}
                    Question: {question}
                    Answer:
                """

                # Create a prompt template
                prompt = PromptTemplate(template=template, input_variables=[
                    "context", "question"])

                # Create embeddings for the pages,
                # then insert into Chroma database
                embeddings = OpenAIEmbeddings()
                vectordb = Chroma.from_documents(pages, embeddings)
                retriever = vectordb.as_retriever()
                chain_type_kwargs = {"prompt": prompt}

                # Initialize the OpenAI module
                llm = OpenAI(temperature=0)
                # load and run the retrieval QA chain
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs=chain_type_kwargs,
                    verbose=True
                )

                # Loop through the predefined questions and
                # summarize the document based on each
                for question in predefined_questions:
                    search = vectordb.similarity_search(question)
                    answer = chain.run(input_documents=search, query=question)
                    st.markdown(f"{answer}")

            except Exception as e:
                st.write(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
