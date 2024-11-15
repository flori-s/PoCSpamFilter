# Spam Detectie Systeem

Dit project is een webapplicatie voor het detecteren van spam e-mails. De applicatie is gebouwd met Flask en maakt
gebruik van een getraind machine learning model om te voorspellen of een e-mail spam is of niet.

## Functies

- Voorspellen of een e-mail spam is op basis van vier kenmerken:
    - Aantal woorden
    - Hoofdletters
    - Onbetrouwbare afzender
    - Aantal links
- Webinterface voor het invoeren van e-mailkenmerken en het weergeven van voorspellingen
- Modeltraining met behulp van Random Forest Classifier en GridSearchCV voor hyperparameteroptimalisatie

## Live Demo

Je kunt de live demo van dit project bekijken op Render. Geen installatie nodig.

[Live Demo op Render](https://pocspamfilter.onrender.com)

## Bestanden

- `app.py`: Bevat de Flask webapplicatie.
- `Model.py`: Bevat de code voor het trainen van het machine learning model.
- `requirements.txt`: Bevat de lijst van vereiste Python-pakketten.

## Gebruik

Bezoek gewoon de live demo-link en begin de spamfilter te gebruiken.
