import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Generate a larger synthetic dataset
X, y = make_classification(n_samples=5000, n_features=4, n_informative=3, n_redundant=1, random_state=42)

# Convert to DataFrame for consistency with original code
df = pd.DataFrame(X, columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
df['is_spam'] = y

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['is_spam']), df['is_spam'], test_size=0.2,
                                                    random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object with cv=2
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the model with the best parameters
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Test the model and calculate the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def predict_spam(aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links):
    new_data = pd.DataFrame([[aantal_woorden, hoofdletters, afzender_onbetrouwbaar, aantal_links]],
                            columns=['aantal_woorden', 'hoofdletters', 'afzender_onbetrouwbaar', 'aantal_links'])
    prediction = model.predict(new_data)
    return prediction[0]
