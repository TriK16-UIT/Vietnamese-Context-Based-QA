from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

qa_pipeline = pipeline("question-answering", model="models/xlm-roberta-base", device="cuda")

context = ""

@app.route("/set_context", methods=["POST"])
def set_context():
    global context
    context = request.json.get("context", "")
    return jsonify({"message": "Context set successfully!"})

@app.route("/ask", methods=["POST"])
def ask():
    global context
    question = request.json.get("question", "")
    if question.lower() == "reset":
        context = ""
        return jsonify({"message": "Context has been reset."})
    if not context:
        return jsonify({"error": "No context set. Please set context first!"}), 400
    result = qa_pipeline(question=question, context=context)
    return jsonify({"answer": result["answer"]})

if __name__ == "__main__":
    app.run(debug=False)