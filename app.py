from flask import Flask, render_template, request, jsonify
from model.phobert import Phobert
from model.xlm_roberta_base import XLMRobertaBase

app = Flask(__name__)

phobert = Phobert()
xlm_roberta = XLMRobertaBase()

@app.route('/')
def hello_world():  # put application's code here
    return render_template(r'index.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.json
    message = data['message']
    model = data['model']
    if model == 'Phobert':
        results = phobert.generate_question(message)
        bot_response = "Phobert generated:\n"
        for idx, question in enumerate(results):
            bot_response += f"{idx + 1}.{question}" + "\n"
    else:
        results = xlm_roberta.generate_question(message)
        bot_response = "XLM Roberta generated:\n"
        for idx, question in enumerate(results):
            bot_response += f"{idx + 1}.{question}\n"

    response = bot_response.replace('_', ' ')
    return jsonify({'message': response})

if __name__ == '__main__':
    app.run()
