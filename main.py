from flask import Flask, request, jsonify
import openai
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

    if 'audio' not in request.files:
        return jsonify({"error": "Missing 'audio' file in form data."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == "":
        return jsonify({"error": "Empty audio file received."}), 400

    # Save the uploaded audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
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
