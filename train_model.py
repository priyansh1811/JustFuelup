import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Data
data = pd.read_csv('Book1.csv')

# Remove leading spaces from all column names
data.columns = data.columns.str.strip()

# Label Encoding for categorical features
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Season'] = le.fit_transform(data['Season'])
data['Destination Type'] = le.fit_transform(data['Destination Type'])
data['Suggestion'] = le.fit_transform(data['Suggestion'])

# Feature Selection
X = data[['Age', 'Gender', 'Season', 'Destination Type']]
y = data['Suggestion']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

