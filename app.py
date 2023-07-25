
from flask import Flask, request, render_template
from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.formatters import HtmlFormatter
from qasyatem import DocumentRetrievalQA

app = Flask(__name__)
# https://pygments.org/styles/
style = 'friendly'

pdf_folder_path = "documents"
huggingface_token = "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"
model_repo_id = "google/flan-t5-large"

qa_system = DocumentRetrievalQA(pdf_folder_path, huggingface_token, model_repo_id)

@app.route("/", methods=["GET"])
def show_panel():
    return render_template("./index.html")

@app.route("/", methods=["POST"])
def completion():
    highighter_style = HtmlFormatter(style=style).get_style_defs('.highlight')
    prompt = request.form["prompt"]
    response = qa_system.ask_question(prompt)
    print(response)
    return render_template("./index.html", response=format(response), highighter_style=highighter_style)



def format(response):
    formatted_response = []
    for document in response.get("documents", []):
        text = document.get("source", {}).get("content", "")
        formatted_response.extend(format_text(text))
    return formatted_response

def format_text(text):
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        if line.startswith("```"):
            formatted_lines.append({"text": line[3:], "code": True})
        else:
            formatted_lines.append({"text": line, "code": False})
    return formatted_lines




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)