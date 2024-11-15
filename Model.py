# Importeer benodigde libraries
# Pandas voor data manipulatie, Torch voor machine learning, en scikit-learn voor preprocessing en evaluatie
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1. Dataset genereren
# Hier maken we een synthetische dataset met behulp van `make_classification`.
# Deze dataset simuleert kenmerken van e-mails (zoals aantal woorden, hoofdletters) en een doelvariabele (is_spam).
X, y = make_classification(
    n_samples=1000,  # 1000 rijen (observaties)
    n_features=4,  # 4 kenmerken per observatie
    n_informative=3,  # 3 kenmerken zijn belangrijk voor classificatie
    n_redundant=1,  # 1 kenmerk bevat overlap met anderen
    random_state=42  # Zorgt voor reproduceerbaarheid
)

# Zet de gegenereerde data in een Pandas DataFrame voor eenvoudiger manipulatie.
# De kolomnamen zijn gekozen op basis van veelgebruikte kenmerken bij spamdetectie.
df = pd.DataFrame(X, columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
df['is_spam'] = y  # Voeg de doelvariabele (is_spam) toe

# 2. Splitsen van de dataset
# Verdeel de gegevens in trainings- en testsets (80% trainen, 20% testen).
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['is_spam']),  # Kenmerken
    df['is_spam'],  # Doelvariabele
    test_size=0.2,  # 20% van de data wordt gebruikt voor testen
    random_state=42  # Reproduceerbaarheid
)

# 3. Data standaardiseren
# Voor machine learning-algoritmes is het belangrijk om data te standaardiseren zodat alle kenmerken hetzelfde schaalniveau hebben.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Pas standaardisatie toe op de trainingsdata
X_test = scaler.transform(X_test)  # Standaardiseer ook de testdata met dezelfde scaler

# 4. Data omzetten naar PyTorch tensors
# PyTorch werkt met tensors als input, dus we zetten onze gestandaardiseerde data om.
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Maak het 2D
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)  # Maak het 2D


# 5. Het neurale netwerk definiëren
# Dit eenvoudige neuraal netwerk heeft:
# - 1 verborgen laag met 16 neuronen
# - ReLU als activatiefunctie voor de verborgen laag
# - Sigmoid als activatiefunctie voor de outputlaag (voor binaire classificatie)
class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # Invoer: 4 kenmerken, Uitvoer: 16 neuronen
        self.fc2 = nn.Linear(16, 1)  # Invoer: 16 neuronen, Uitvoer: 1 output (is_spam)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activatiefunctie voor binaire classificatie

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activatiefunctie
        x = self.fc2(x)  # Lineaire combinatie
        x = self.sigmoid(x)  # Sigmoid activatiefunctie
        return x


# 6. Model initialiseren
# Initialiseer het model, de verliesfunctie (Binary Cross-Entropy Loss), en de optimizer (Adam).
model = SpamClassifier()
criterion = nn.BCELoss()  # Verliesfunctie voor binaire classificatie
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimaliseer met een leersnelheid van 0.001

# 7. Het model trainen
# Train het model in 100 epochs. Bij elke epoch berekenen we het verlies, doen we backpropagation, en passen we de gewichten aan.
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Zet het model in training-modus
    optimizer.zero_grad()  # Reset de gradienten van de vorige iteratie
    outputs = model(X_train)  # Voorspel de output voor de trainingsdata
    loss = criterion(outputs, y_train)  # Bereken het verlies
    loss.backward()  # Backpropagation: bereken de gradienten
    optimizer.step()  # Pas de gewichten aan
    print(f"Epoch [{epoch + 1}/{num_epochs}], Verlies: {loss.item():.4f}")

# 8. Het model en de scaler opslaan
# Sla het getrainde model en de scaler op voor toekomstig gebruik.
torch.save(model.state_dict(), 'spam_model.pth')  # Sla alleen de gewichten op
joblib.dump(scaler, 'scaler.pkl')  # Sla de scaler op

# 9. Model evalueren
# Evalueer de prestaties van het model op de testset.
model.eval()  # Zet het model in evaluatiemodus
with torch.no_grad():  # Geen gradienten nodig
    y_pred = model(X_test)  # Voorspel de output
    y_pred = (y_pred > 0.5).float()  # Converteer waarschijnlijkheden naar binaire waarden
    accuracy = accuracy_score(y_test, y_pred)  # Bereken de nauwkeurigheid
    conf_matrix = confusion_matrix(y_test, y_pred)  # Confusiematrix
    class_report = classification_report(y_test, y_pred)  # Classificatierapport

# Toon evaluatieresultaten
print(f"Model nauwkeurigheid: {accuracy}")
print("Confusiematrix:")
print(conf_matrix)
print("\nClassificatieverslag:")
print(class_report)

# Sla de nauwkeurigheid op in een tekstbestand
with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))


# 10. Nieuwe voorspellingen maken
# Definieer een functie waarmee nieuwe e-mails geclassificeerd kunnen worden.
def predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links):
    try:
        # Laad het getrainde model en de scaler
        spam_model = SpamClassifier()
        spam_model.load_state_dict(torch.load('spam_model.pth'))
        spam_model.eval()  # Zet het model in evaluatiemodus
        data_scaler = joblib.load('scaler.pkl')

        # Bereid de nieuwe gegevens voor
        new_data = pd.DataFrame([[aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links]],
                                columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
        new_data = data_scaler.transform(new_data)  # Standaardiseer de nieuwe gegevens
        new_data = torch.tensor(new_data, dtype=torch.float32)  # Zet om naar een tensor

        # Debugging: toon de ingevoerde gegevens
        print(f"Invoergegevens: {new_data}")

        # Voorspel of de nieuwe e-mail spam is
        with torch.no_grad():
            prediction = spam_model(new_data)  # Voorspel de output
            prediction_prob = prediction.item()  # Verkrijg de kans dat het spam is
            prediction = float(prediction_prob > 0.75)  # Gebruik een drempelwaarde van 0.75

        # Debugging: toon de voorspelde waarschijnlijkheid
        print(f"Voorspelde waarschijnlijkheid: {prediction_prob}")

        # Retourneer het resultaat als 'Spam' of 'Geen Spam'
        return 'Spam' if prediction == 1 else 'Geen Spam'

    except Exception as e:
        # Retourneer een foutmelding als er iets misgaat
        return f"Er is een fout opgetreden: {e}"
