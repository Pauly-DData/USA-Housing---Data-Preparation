import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the dataset
df_2 = pd.read_csv(r"/kaggle/input/usa-housing/USA_Housing.csv")

# Split data into two sets:
# X contains the independent column 'Avg. Area Income',
# y contains the target/outcome variable 'Price'.
X = df_2[['Avg. Area Income']]
y = df_2['Price']

# Display the head of the dataframe (optional)
print(df_2.head())

# Partition data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the test dataset
predictions = model.predict(X_test)

# Display model metrics
print('coefficient of determination:', model.score(X_train, y_train))  # 𝑅²
print('intercept:', model.intercept_)  # represents coefficient 𝑏₀
print('slope:', model.coef_)  # represents coefficient 𝑏₁

# Graphical illustration
plt.scatter(X_train, y_train, color='g')
plt.plot(X_train, model.predict(X_train), color='k')
plt.show()
