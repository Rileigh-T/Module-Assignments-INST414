import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('imdb_top_1000.csv')
df.replace({'\\N': np.nan}, inplace=True)

df['Runtime_minutes'] = df['Runtime'].str.extract('(\d+)').astype(float)
df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')

df['Runtime_minutes'].fillna(df['Runtime_minutes'].median(), inplace=True)
df['Meta_score'].fillna(df['Meta_score'].median(), inplace=True)
df['No_of_Votes'].fillna(df['No_of_Votes'].median(), inplace=True)

X = df[['Runtime_minutes', 'Meta_score', 'No_of_Votes']]
y = df['IMDB_Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

predictions = pd.DataFrame({
    'Movie': df.loc[X_test.index, 'Series_Title'],
    'Actual': y_test,
    'Predicted': y_pred,
    'Error': abs(y_test - y_pred)
})
worst_predictions = predictions.sort_values('Error', ascending=False).head(5)
print("\n5 Movies with Worst Prediction Errors:")
print(worst_predictions)

plt.figure(figsize=(10, 4))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table_data = [
    [movie, f"{actual:.1f}", f"{pred:.1f}", f"{error:.1f}"] 
    for movie, actual, pred, error in zip(
        worst_predictions['Movie'],
        worst_predictions['Actual'],
        worst_predictions['Predicted'],
        worst_predictions['Error']
    )
]

table = plt.table(
    cellText=table_data,
    colLabels=["Movie Title", "Actual Rating", "Predicted Rating", "Error"],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Top 5 Movies with Largest Prediction Errors")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Actual IMDb Rating")
plt.ylabel("Predicted IMDb Rating")
plt.title("Actual vs. Predicted IMDb Ratings")
plt.tight_layout()
plt.show()
