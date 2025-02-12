import subprocess
import openai
import gradio as gr
import config
openai.api_key = config.OPENAI_API_KEY

messages=[
            {"role": "system", "content": "You are a helpful assistant."}
            ]

def speech_to_text(audio_file):
    '''Transcribe an audio file to text using OpenAI's whisper model.

    Args:
        audio (str): Path to the audio file to transcribe.

    Returns:
        str: The transcribed text.

    '''

    if audio_file is None:
        return "No audio detected. Please try again."

    with open(audio_file, "rb") as file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )

    return transcription.text

def query_ChatGPT(messages):
    '''Send messages to OpenAI's Chat GPT-3 model and get response.

    Args:
        messages (list): A list of messages to send to the model.

    Returns:
        str: The response message from the model.

    '''
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
        )
    response_message = response['choices'][0]['message']['content']
    return response_message

def create_chat_transcript(messages):
    '''Create a chat transcript from all non-system messages.

    Args:
        messages (list): A list of messages, where each message is a dictionary
            with keys 'role' and 'content'.

    Returns:
        str: The chat transcript, where each message is formatted as
            "<role>: <content>\n\n".

    '''
    chat_transcript = ''
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += f"{message['role']}: {message['content']} \n\n"
    return chat_transcript

def greet(audio):
    global messages
    transcript = speech_to_text(audio)
    messages.append({"role": "user", "content": transcript})
    response = query_ChatGPT(messages)
    subprocess.call(["say", response])
    messages.append({"role": "assistant", "content": response})
    chat_transcript = create_chat_transcript(messages)

    return chat_transcript

demo = gr.Interface(
    fn=greet,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text"
)

demo.launch()
