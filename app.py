import os 
from pypdf import PdfReader

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import openai
from openai import OpenAI

from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import streamlit as st
st.set_page_config(
    page_title="Research Copilot",
    page_icon="🤖",
    layout="wide",
)



# customization
api_key = st.secrets["OPENAI_API_KEY"]
MODEL = 'gpt-4o-mini'
MAX_TOKEN= 300
TEMPERATURE = 0.2

run_tab1 = True

os.environ['OPENAI_API_KEY'] = api_key

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  organization='org-4m59LIFJeSjBW5H3Y1nsPlB5',
  project='proj_oSJbtQ2Je7QkjUsnP9RY681d',
  api_key=os.getenv("OPENAI_API_KEY")
)

# Load summarization model
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

summarizer = load_summarizer()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)


    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to divide text into sections
def identify_sections(text):
    sections = {
        "Abstract": text.split("Introduction")[0],
        # "Introduction": text.split("Introduction")[1].split("Methodology")[0],
        # if "Methodology" in text:
        #     "Methodology": text.split("Methodology")[1].split("Results")[0],
        # "Results": text.split("Results")[1].split("Conclusion")[0],
        "Conclusion": text.split("Conclusion")[1].split("References")[0],
    }
    return sections

# Summarize each section
def summarize_sections(sections):
    summary = {}
    for section, content in sections.items():
        # print(content)
        if len(content) > 50:
            # print(summarizer(content, max_length=150, min_length=50, do_sample=False))
            summary_text = summarizer(content, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            # print(summary_text)
            summary[section] = summary_text
        else:
            summary[section] = content
    return summary

# Function to extract terminology and definitions
def get_terminology_definitions(text):
    prompt = f"Extract complex terms and their definitions from the following text:\n\n{text}\n\nProvide terms and definitions. Limit your response to maximum 200 words"
    response = client.chat.completions.create(
        model= MODEL,
        messages=[
        {"role": "system", "content": "You are a 15 years experienced professor."},
        {
            "role": "user",
            "content": prompt
        }   
    ],
        max_tokens=MAX_TOKEN,
        temperature=TEMPERATURE
    )
    # print (response.choices[0].message)
    return response.choices[0].message.content

# Function to generate key notes for the paper
def generate_key_notes(text):
    prompt = f"Generate main points to keep in my notes from the following text:\n\n{text}. Limit your response to maximum 200 words"
    response = client.chat.completions.create(
        model= MODEL,
        messages=[
        {"role": "system", "content": "You are a 15 years experienced professor."},
        {
            "role": "user",
            "content": prompt
        } 
    ],
        max_tokens=MAX_TOKEN,
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content

# Function to explain paper with example
def explain_with_example(text):
    prompt = f"Explain the main concepts of the following text with a practical example:\n\n{text}. Limit your response to maximum 200 words"
    response = client.chat.completions.create(
        model= MODEL,
        messages=[
        {"role": "system", "content": "You are a 15 years experienced professor."},
        {
            "role": "user",
            "content": prompt
        } 
    ],
        max_tokens=MAX_TOKEN,
        temperature= TEMPERATURE
    )
    return response.choices[0].message.content

# Q&A functionality with memory
chat_history_track = []

def research_paper_qa_with_memory(question, context):
    chat_history_track.append({"role": "user", "content": question})
    messages = [{"role": "system", "content": "You are an expert teacher who answers questions based on the following research paper context. Limit your response to maximum 200 words"}]
    for history in chat_history_track:
        messages.append(history)
    messages.append({"role": "assistant", "context": context})
    
    response = client.chat.completions.create(
        model= MODEL,
        messages=messages,
        max_tokens=MAX_TOKEN,
        temperature=TEMPERATURE
    )
    answer = response.choices[0].message.content
    chat_history_track.append({"role": "assistant", "content": answer})
    return answer


def createEmbeddings():
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002" ,
    chunk_size=1   
    )
    return embeddings

def save_db(doc, embeddings):

    pdf_loader = PyPDFLoader('./temp.pdf') 

    loaders = [pdf_loader]
    # print(loaders)
    documents = []

    for loader in loaders:
        # print(loader.load())
        documents.extend(loader.load())

    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap = 100)
    all_documents = text_spliter.split_documents(documents)

    print(f"Total number of documents: {len(all_documents)}")

    batch_size = 96

    # Calculate the number of batches
    num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

    texts = ["FAISS is an important library", "Langchain supports FAISS"]
    db = FAISS.from_texts(texts, embeddings)
    retv = db.as_retriever()

    # Iterate over batches
    for batch_num in range(num_batches):
        # Calculate start and end indices for the current batch
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        # Extract documents for the current batches
        batch_documents = all_documents[start_index:end_index]
        # Yout code to process each document goes here
        retv.add_documents(batch_documents)
        print("start and end: ", start_index, end_index)
    db.save_local("faiss_index")

# Streamlit app layout

st.header(":blue[Research Copilot]", divider = True)
st.write(":violet[Understand your research paper within 5 minutes!]")



pdf_file = st.file_uploader("Upload PDF", type=["pdf"])






tab1, tab2= st.tabs(["Summary", "QnA"])



if pdf_file is not None:
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(pdf_file.getvalue())
        file_name = pdf_file.name

    loader = PyPDFLoader(temp_file)
    documents = loader.load_and_split()
    
    with tab1:
        if run_tab1:
            with st.spinner("Extracting text from PDF..."):
                full_text = extract_text_from_pdf(pdf_file)

            with st.spinner("Identifying sections..."):
                sections = identify_sections(full_text)

            with st.spinner("Summarizing sections..."):
                summaries = summarize_sections(sections)
                # summaries = summarize(full_text)
            
            # Display summary by sections
            st.header(":green[Sections Summary]", divider=True)
            for section, summary in summaries.items():
                st.subheader(section)
                st.write(summary)

            # col1, col2 = st.columns(2)
            # with col1:
               
                    # Display terminology
            st.header(":green[Complex Terminologies]", divider=True)
            terminology_definitions = get_terminology_definitions(full_text)
            st.write(terminology_definitions)
            # with col2:
               
                # Generate key notes
            st.header(":green[Key Notes]", divider=True)
            key_notes = generate_key_notes(full_text)
            st.write(key_notes)

    with tab2:
        # Interactive Q&A with memory
        st.header(":green[Ask Questions]")        
        embeddings = createEmbeddings()

        save_db(pdf_file, embeddings)


        db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        retv = db.as_retriever(search_type="similarity", seach_kwargs = {"k": 5})

        llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

        history = StreamlitChatMessageHistory(key="chat_messages")
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=history, return_messages=True)
        #You are an expert teacher who answers questions based on the following research paper context. Limit your response to maximum 200 words
        template = """You are an AI chatbot having a conversation with a human.
            Human: {question}
            AI: """

        prompt = PromptTemplate(input_variables=['question'], template=template)

        
        from langchain.prompts import (
                ChatPromptTemplate,
                HumanMessagePromptTemplate,
                SystemMessagePromptTemplate,
            )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "Your name is AI, you are a nice virtual assistant. The context is:\n{context}"
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "{question}"
        )

        qa = ConversationalRetrievalChain.from_llm(llm, retriever = retv, memory= memory, return_source_documents = False)

        # print(history.messages)
        for msg in history.messages:
            print(msg.type)
            st.chat_message(msg.type).write(msg.content)
            
        # if question:
        if x := st.chat_input(placeholder="ask your question.."):
            run_tab1 = False
            st.chat_message('human').write(x)
            with st.spinner("Generating answer..."):
                
                answer = qa.invoke({"question": x})
                # answer = llm_chain.run(x)
                st.chat_message("ai").write(answer['answer'])

       

        # Clear chat history
        if st.button("Clear Conversation"):
            # st.session_state.chat_history = []
            st.session_state.conversation = None
            st.session_state.chat_history_track = None    
            memory.clear()

# st.markdown(":violet[Made at Hackathon @ HackUNT | University of North Texas]", unsafe_allow_html=True)




footer_html = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: lightblue;
            color: black;
            text-align: center;
        }
    </style>

    <div class="footer">
    <p>Made at Hackathon @ HackUNT | University of North Texas | All rights reserved</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
