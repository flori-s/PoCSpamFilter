# app.py
from os import environ
from flask import Flask, render_template, request
from Model import predict_spam, accuracy

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get user input
            aantal_woorden = int(request.form['aantal_woorden'])
            hoofdletters = int(request.form['hoofdletters'])
            afzender_onbetrouwbaar = int(request.form['afzender_onbetrouwbaar'])
            aantal_links = int(request.form['aantal_links'])

            # Validate input
            if hoofdletters not in [0, 1] or afzender_onbetrouwbaar not in [0, 1]:
                error_message = "Ongeldige invoer. Gebruik 0 of 1 voor hoofdletters en afzender."
            else:
                # Make prediction
                prediction = predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links)
                result = "Spam" if prediction == 1 else "Geen Spam"

        except Exception as e:
            error_message = f"Er is een fout opgetreden: {str(e)}"

    return render_template('index.html', result=result, error_message=error_message, accuracy=accuracy)


if __name__ == '__main__':
    port = int(environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port)
