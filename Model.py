import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


# Generate a larger synthetic dataset
X, y = make_classification(n_samples=2000, n_features=4, n_informative=3, n_redundant=1, random_state=42)

# Convert to DataFrame for consistency with original code
df = pd.DataFrame(X, columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
df['is_spam'] = y

# Check data balance
print(df['is_spam'].value_counts())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['is_spam']), df['is_spam'], test_size=0.2,
                                                    random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid with more options
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

# Initialize the GridSearchCV object with cv=5
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the model with the best parameters
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'spam_model.pkl')

# Test the model and calculate the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy}")

# Evaluate model performance
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Check feature importance
feature_importances = model.feature_importances_
features = ['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links']
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print("\nFeature Importances:")
print(importance_df)


def predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links):
    try:
        # Load the trained model
        loaded_model = joblib.load('spam_model.pkl')

        # Prepare the new data
        new_data = pd.DataFrame([[aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links]],
                                columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
        # Standardize the new data
        new_data = scaler.transform(new_data)
        print(new_data)
        # Make prediction
        prediction = loaded_model.predict(new_data)

        # Return the prediction result
        return 'Spam' if prediction[0] == 1 else 'Not Spam'

    except Exception as e:
        return f"An error occurred: {e}"
