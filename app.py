
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
    highlighter_style = HtmlFormatter(style=style).get_style_defs('.highlight')
    prompt = request.form["prompt"]
    response = qa_system.ask_question(prompt)
    result = response.get('result', '')  # Extract the 'result' key from the response
    return render_template("./index.html", response=result, highlighter_style=highlighter_style)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)