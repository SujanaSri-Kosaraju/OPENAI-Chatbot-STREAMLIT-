# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# extracting text from pdf
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# generating conversation chain
def get_conversationchain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer')  # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory)
    return conversation_chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    if question:
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response["chat_history"]
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg.content,), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Pdf Bot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # State variable to manage view transitions
    view_state = st.session_state.get("view_state", "select_pdf")

    # Page 1: Left side - File upload and process
    if view_state == "select_pdf":
        st.header("Interact with multiple documents :books:")
        with st.form(key='pdf_upload_form'):
            docs = st.file_uploader("Upload your PDF", accept_multiple_files=True, type=["pdf"])
            submit_button = st.form_submit_button("Process")

        if submit_button:
            with st.spinner("Processing"):
                # get the pdf
                raw_text = get_pdf_text(docs)

                # get the text chunks
                text_chunks = get_chunks(raw_text)

                # create vectorstore
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversationchain(vectorstore)

                # Transition to the next view
                view_state = "enter_question"
                st.session_state.view_state = view_state

                # Trigger page reload to show the second view
                st.experimental_rerun()

    # Page 2: Right side - Text input, question handling, and back icon
    elif view_state == "enter_question":
        st.header("Interact with multiple documents :books:")

        # Back icon to navigate back to the first view
        if st.button("⬅️ Back"):
            view_state = "select_pdf"
            st.session_state.view_state = view_state
            st.experimental_rerun()

        st.subheader("Ask your query from the PDF document:")
        question = st.text_input("Ask your query from the PDF document:")

        # Form submit button to handle Enter key press
        if st.form(key='question_form'):
            handle_question(question)

        # if st.form_submit_button(""):
        #     handle_question(question)

    # Save the current view state
    st.session_state.view_state = view_state


if __name__ == '__main__':
    main()
