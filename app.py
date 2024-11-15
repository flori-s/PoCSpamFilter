import os
from flask import Flask, render_template, request
from Model import predict_spam

# Maak een Flask-webapplicatie aan
app = Flask(__name__)

# Read the accuracy from the file
with open('accuracy.txt', 'r') as f:
    accuracy = float(f.read())

# Route voor de hoofdpagina, ondersteunt zowel GET als POST verzoeken
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_message = None

    # Als het verzoek een POST is, verwerk dan de gebruikersinvoer
    if request.method == 'POST':
        try:
            # Haal de gebruikersinvoer op uit het formulier
            aantal_woorden = int(request.form['aantal_woorden'])
            hoofdletters = int(request.form['hoofdletters'])
            afzender_onbetrouwbaar = int(request.form['afzender_onbetrouwbaar'])
            aantal_links = int(request.form['aantal_links'])

            # Valideer de invoer: hoofdletters en afzender_onbetrouwbaar moeten 0 of 1 zijn
            if hoofdletters not in [0, 1] or afzender_onbetrouwbaar not in [0, 1]:
                error_message = "Ongeldige invoer. Gebruik 0 of 1 voor hoofdletters en afzender."
            else:
                # Maak een voorspelling voor spam of geen spam
                prediction = predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links)

                # Toon "Spam" of "Geen Spam" op basis van de voorspelling
                result = "Spam" if prediction == "Spam" else "Geen Spam"

        except Exception as e:
            # Als er een fout optreedt, geef deze weer
            error_message = f"Er is een fout opgetreden: {str(e)}"

    # Render de HTML-pagina en geef de resultaten weer
    return render_template('index.html', result=result, error_message=error_message, accuracy=accuracy)

# Start de Flask-applicatie
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))  # Haal de poort op vanuit omgevingsvariabelen of gebruik 5000
    app.run(host='0.0.0.0', port=port)  # Start de app en laat deze beschikbaar zijn op alle netwerkinstellingen (0.0.0.0)