import os
#from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from assistant import (
    assistant_response,
    load_json_data,
    send_user_message,
    start_conversation,
    extract_cpf_and_dob,
    search_cpf_and_dob,
    save_messages_to_file
)

#load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# KINDER_ASSISTANT_ID = os.getenv("KINDER_ASSISTANT_ID")
# AGGRESSIVE_ASSISTANT_ID = os.getenv("AGGRESSIVE_ASSISTANT_ID")
# FORMAL_ASSISTANT_ID = os.getenv("FORMAL_ASSISTANT_ID")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
KINDER_ASSISTANT_ID = st.secrets["KINDER_ASSISTANT_ID"]
AGGRESSIVE_ASSISTANT_ID = st.secrets["AGGRESSIVE_ASSISTANT_ID"]
FORMAL_ASSISTANT_ID = st.secrets["FORMAL_ASSISTANT_ID"]

st.set_page_config(
    page_title="AcerBot - Assistente Virtual",
    page_icon="icon.png", 
    layout="wide"
)

st.image("icon.png", width=90)
st.title("AcerBot")
st.markdown("Assistente Virtual que facilita a negociação de dívidas de débitos")

# Sidebar for assistant type selection and control buttons
st.sidebar.title("Configurações")
assistant_type = st.sidebar.selectbox(
    "Escolha o tipo de assistente:",
    ["Gentil", "Agressivo", "Formal"],
    index=0
)
if st.sidebar.button("Reiniciar"):
    st.session_state.clear()
    st.session_state.messages = []
    thread = start_conversation(" ")
    st.session_state.thread_id = thread.id
    st.session_state.cpf = None
    st.session_state.dob = None
    st.rerun()

if st.sidebar.button("Salvar"):
    filename = save_messages_to_file(st.session_state.messages)
    st.sidebar.success(f"Chat salvo em {filename}")

# Map the assistant type to the corresponding ASSISTANT_ID
if assistant_type == "Gentil":
    ASSISTANT_ID = KINDER_ASSISTANT_ID
elif assistant_type == "Agressivo":
    ASSISTANT_ID = AGGRESSIVE_ASSISTANT_ID
elif assistant_type == "Formal":
    ASSISTANT_ID = FORMAL_ASSISTANT_ID

client = OpenAI(api_key=OPENAI_API_KEY)
json_data = load_json_data('database/case_cientista_de_dados_ia.json')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    thread = start_conversation(" ")
    st.session_state.thread_id = thread.id

# Initialize CPF and DOB tracking
if "cpf" not in st.session_state:
    st.session_state.cpf = None
if "dob" not in st.session_state:
    st.session_state.dob = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_message := st.chat_input("Digite algo..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Extract CPF and DOB from user message
    cpf, dob = extract_cpf_and_dob(st.session_state.messages)
    if cpf and dob:
        st.session_state.cpf = cpf
        st.session_state.dob = dob
    
    # Send user message to assistant
    send_user_message(st.session_state.thread_id, user_message)
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_message)

    # Prepare additional instructions if CPF and DOB are found
    additional_instructions = ""
    if st.session_state.cpf and st.session_state.dob:
        instructions = search_cpf_and_dob(
            json_data, 
            st.session_state.cpf,
            st.session_state.dob
        )
        additional_instructions = instructions

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = assistant_response(
            st.session_state.thread_id, 
            ASSISTANT_ID, 
            instructions=additional_instructions
        )
        response = st.markdown(stream)

    st.session_state.messages.append({"role": "assistant", "content": stream})