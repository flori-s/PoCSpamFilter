import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, n_redundant=1, random_state=42)

# Convert to DataFrame for consistency
df = pd.DataFrame(X, columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
df['is_spam'] = y

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['is_spam']), df['is_spam'], test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the neural network model
class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model, loss function, and optimizer
model = SpamClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model and scaler
torch.save(model.state_dict(), 'spam_model.pth')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

print(f"Model accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save the accuracy to a file
with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Define the predict_spam function
def predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links):
    try:
        # Load the trained model and scaler
        spam_model = SpamClassifier()
        spam_model.load_state_dict(torch.load('spam_model.pth', weights_only=True))
        spam_model.eval()
        data_scaler = joblib.load('scaler.pkl')

        # Prepare the new data
        new_data = pd.DataFrame([[aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links]],
                                columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
        new_data = data_scaler.transform(new_data)
        new_data = torch.tensor(new_data, dtype=torch.float32)

        # Debugging information
        print(f"Input data: {new_data}")

        # Make prediction
        with torch.no_grad():
            prediction = spam_model(new_data)
            prediction_prob = torch.sigmoid(prediction).item()  # Get the probability
            prediction = float(prediction_prob > 0.8)

        # Debugging information
        print(f"Prediction probability: {prediction_prob}")

        # Return the prediction result
        return 'Spam' if prediction == 1 else 'Not Spam'

    except Exception as e:
        return f"An error occurred: {e}"