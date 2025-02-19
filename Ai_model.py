import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


csv_file = "dataa.csv" 
data = pd.read_csv(csv_file)


print(data.head())

print(data.isnull().sum())

data = data.fillna(0)


numerical_features = ['feature1', 'feature2'] 
if all(col in data.columns for col in numerical_features):
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])


categorical_columns = ['category_column']  
data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns])

# Define target variable (ensure this column exists)
target_column = 'target_column'  # Update with actual target column
if target_column in data.columns:
    X = data.drop(target_column, axis=1)
    y = data[target_column]
else:
    raise ValueError(f"Target column '{target_column}' not found in dataset")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
