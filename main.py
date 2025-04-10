from flask import Flask, request, jsonify
import openai
import base64
import tempfile
import os

# Set your OpenAI key here or via environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
messages = [{"role": "system", "content": "You are a helpful assistant."}]

@app.route("/", methods=["GET"])
def health_check():
    return "Whisper + GPT backend is running."

@app.route("/transcribe-chat", methods=["POST"])
def transcribe_chat():
    global messages
    data = request.get_json()

    if not data or 'audio_base64' not in data:
        return jsonify({"error": "Missing audio_base64 field."}), 400

    # Decode audio
    audio_data = base64.b64decode(data['audio_base64'])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data)
        tmp.flush()
        audio_path = tmp.name

    try:
        # Transcribe with Whisper
        with open(audio_path, "rb") as file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
        user_input = transcription.text
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)

    # ChatGPT response
    messages.append({"role": "user", "content": user_input})
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "user": user_input,
        "assistant": assistant_reply
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
