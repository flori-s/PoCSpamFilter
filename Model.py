import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Genereer een synthetische dataset met 2000 voorbeelden
# We hebben 4 kenmerken (features) en een binaire target variabele (spam of niet-spam)
X, y = make_classification(n_samples=2000, n_features=4, n_informative=3, n_redundant=1, random_state=42)

# Zet de gegenereerde features om in een pandas DataFrame voor betere leesbaarheid en consistentie
df = pd.DataFrame(X, columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
df['is_spam'] = y  # Voeg de target variabele toe (1 voor spam, 0 voor niet-spam)

# Verdeel de dataset in een trainingsset en een testset (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['is_spam']), df['is_spam'], test_size=0.2,
                                                    random_state=42)

# Standaardiseer de gegevens (meestal nodig voor machine learning modellen)
# Dit zorgt ervoor dat alle kenmerken dezelfde schaal hebben
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Pas de standaardisatie toe op de trainingsset
X_test = scaler.transform(
    X_test)  # Pas de standaardisatie toe op de testset (gebruik dezelfde schaal als bij de trainingsset)

# Definieer de parameter grid voor GridSearch (dit zijn de hyperparameters die we willen optimaliseren)
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Aantal bomen in het Random Forest
    'max_depth': [None, 10, 20, 30, 40],  # Maximale diepte van de bomen
    'min_samples_split': [2, 5, 10, 15],  # Minimum aantal monsters (data) vereist om een interne splitsing te maken
    'min_samples_leaf': [1, 2, 4, 6]  # Minimum aantal monsters vereist in een blad van de boom
}

# Initialiseer GridSearchCV voor het optimaliseren van de hyperparameters met behulp van cross-validatie (cv=5 betekent 5-voudige cross-validatie)
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit de GridSearch op de trainingsdata (dit betekent dat de grid search de beste combinatie van hyperparameters zoekt)
grid_search.fit(X_train, y_train)

# Verkrijg de beste hyperparameters en de beste score (de hoogste gemiddelde score over de cross-validatie)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train het Random Forest model met de beste hyperparameters
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Sla het getrainde model op in een bestand met behulp van joblib
joblib.dump(model, 'spam_model.pkl')

# Test het model op de testset en bereken de nauwkeurigheid
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Evalueer de modelprestaties
conf_matrix = confusion_matrix(y_test, y_pred)  # Confusiematrix voor de prestatieanalyse
class_report = classification_report(y_test,
                                     y_pred)  # Gedetailleerd classificatierapport met precisie, recall en F1-score

# Bekijk de feature importances (hoe belangrijk elk kenmerk is voor de voorspelling)
feature_importances = model.feature_importances_
features = ['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links']
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})


# Functie om spam te voorspellen op basis van nieuwe gegevens
def predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links):
    try:
        # Laad het getrainde model uit het bestand
        loaded_model = joblib.load('spam_model.pkl')

        # Maak een DataFrame voor de nieuwe gegevens die we willen voorspellen
        new_data = pd.DataFrame([[aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links]],
                                columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])

        # Standaardiseer de nieuwe gegevens (gebruik dezelfde scaler die we eerder hebben getraind)
        new_data = scaler.transform(new_data)

        # Maak een voorspelling met het geladen model
        prediction = loaded_model.predict(new_data)

        # Retourneer de voorspelling als 'Spam' of 'Not Spam' afhankelijk van de uitkomst
        return 'Spam' if prediction[0] == 1 else 'Not Spam'

    except Exception as e:
        # Als er een fout optreedt, geef de foutmelding terug
        return f"Er is een fout opgetreden: {e}"