import subprocess
import openai
import gradio as gr
import config
openai.api_key = config.OPENAI_API_KEY

messages=[
            {"role": "system", "content": "You are a helpful assistant."}
            ]

def speech_to_text(audio):
    '''Transcribe audio file to text by whisper.'''
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe('whisper-1', audio_file)
    return transcript['text']

def query_ChatGPT(messages):
    '''Send messages to OpenAI's Chat GPT-3 model and get response.'''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    response_message = response['choices'][0]['message']['content']
    return response_message

def create_chat_transcript(messages):
    '''Create chat transcript from all messages.'''
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

demo = gr.Interface(fn=greet, inputs=gr.Audio(source = 'microphone', type = 'filepath'), outputs="text")

demo.launch()
