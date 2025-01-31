import pandas as pd


# Load the dataset
data = pd.read_csv('datasett')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Get dataset statistics
print(data.describe())
data = data.fillna(0)  # Example: Fill missing values with 0
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])

data = pd.get_dummies(data, columns=['category_column'])

from sklearn.model_selection import train_test_split

X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])