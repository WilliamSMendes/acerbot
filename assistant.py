import os
import re
import json
import uuid
import streamlit as st
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

class EventHandler(AssistantEventHandler):  
  
  @override
  def on_text_created(self, text) -> None:
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    print(f"\nassistant > {tool_call.type}\n", flush=True)
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)

def assistant_response_streaming(thread_id, assistant_id, instructions=""):
    """
    Stream responses from the assistant until the conversation is complete.

    Args:
        thread_id (str): The ID of the thread.
        assistant_id (str): The ID of the assistant.
        instructions (str, optional): Additional instructions for the assistant. Defaults to "".
    """
    event_handler = EventHandler()
    
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()


def start_conversation(initial_message):
    """
    Start a new conversation with an initial message.

    Args:
        initial_message (str): The initial message to start the conversation.

    Returns:
        thread: The created thread object.
    """
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": initial_message
            }
        ]
    )
    return thread


def send_user_message(thread_id, message):
    """
    Send a user message to the specified thread.

    Args:
        thread_id (str): The ID of the thread.
        message (str): The message content.
    """
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )


def assistant_response(thread_id, assistant_id, instructions=""):
    """
    Get a response from the assistant for the specified thread.

    Args:
        thread_id (str): The ID of the thread.
        assistant_id (str): The ID of the assistant.
        instructions (str, optional): Additional instructions for the assistant. Defaults to "".

    Returns:
        str: The content of the assistant's response.
    """
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id, assistant_id=assistant_id, additional_instructions=instructions
    )
    messages = list(client.beta.threads.messages.list(thread_id=thread_id, run_id=run.id))
    print(messages)
    message_content = messages[0].content[0].text
    return message_content.value


def load_json_data(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def extract_cpf_and_dob(messages, role='assistant'):
    """
    Extract CPF and date of birth from the list of messages of a specific role.

    Args:
        messages (list): A list of messages.
        role (str): The role of the messages to filter (default is 'assistant').

    Returns:
        tuple: The CPF and date of birth if found, otherwise (None, None).
    """
    cpf_pattern = r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"
    dob_pattern = r"\b\d{2}/\d{2}/\d{4}\b"

    filtered_messages = [msg for msg in messages if msg.get('role') == role]

    for message in reversed(filtered_messages):
        text = message.get('content', '')
        
        cpf_match = re.search(cpf_pattern, text)
        dob_match = re.search(dob_pattern, text)

        if cpf_match and dob_match:
            cpf = cpf_match.group()
            dob = dob_match.group()
            return cpf, dob

    return None, None


def search_cpf_and_dob(json_data, cpf, dob):
    """
    Search for the CPF and date of birth in the JSON data.

    Args:
        json_data (dict): The JSON data containing user information.
        cpf (str): The CPF to search for.
        dob (str): The date of birth to search for.

    Returns:
        str: A message indicating the result of the search.
    """
    numeric_cpf = cpf.replace('.', '').replace('-', '')
    dob_parts = dob.split('/')
    formatted_dob = f"{dob_parts[2]}-{dob_parts[1]}-{dob_parts[0]}T00:00:00"

    for user in json_data:
        if str(user.get('cpf_cnpj')) == numeric_cpf:
            if user.get('data_nascimento') == formatted_dob:
                return f"This is the client information to use: {user}"
            else:
                return (
                    f"""Say to the client that the CPF {cpf} and the date of birth {dob} weren't found in our system 
                    and ask the client to provide the information again or would prefer you to help with another question"""
                )
    return (
        f"""Say to the client that the CPF {cpf} and the date of birth {dob} weren't found in our system 
        and ask the client to provide the information again or would prefer you to help with another question"""
    )


def save_messages_to_file(messages):
    """
    Save messages to a file.

    Args:
        messages (list): A list of messages.

    Returns:
        str: The filename where the messages are saved.
    """
    os.makedirs("output", exist_ok=True)
    filename = f"output/{uuid.uuid4()}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        for message in messages:
            file.write(f"{message['role']}: {message['content']}\n")
    return filename


def list_messages(thread_id):
    """
    List messages for a given thread ID.

    Args:
        thread_id (str): The ID of the thread.

    Returns:
        list: A list of messages in the thread.
    """
    return client.beta.threads.messages.list(thread_id=thread_id)


def structure_messages(messages):
    """
    Structure messages for better readability.

    Args:
        messages (list): A list of messages.

    Returns:
        list: A list of structured messages.
    """
    structured_messages = []

    for message in reversed(messages.data):
        role = message.role
        for content_block in message.content:
            text_content = content_block.text.value
            structured_message = f"{role}: {text_content}"
            structured_messages.append(structured_message)

    return structured_messages


def write_messages_to_file(structured_messages, file_name):
    """
    Write structured messages to a file.

    Args:
        structured_messages (list): A list of structured messages.
        file_name (str): The name of the file to write to.
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        for message in structured_messages:
            file.write(message + '\n')


def save_conversation_to_file(messages, file_name):
    """
    Save the entire conversation to a file.

    Args:
        messages (list): A list of messages.
        file_name (str): The name of the file to save the conversation to.
    """
    structured_messages = structure_messages(messages)
    file_name = f"output/{file_name}.txt"
    write_messages_to_file(structured_messages, file_name)
