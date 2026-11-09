import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# 1. הורדה וטעינה אוטומטית של הנתונים
path = kagglehub.dataset_download("zafarali27/house-price-prediction-dataset")
df = pd.read_csv(path + "/House Price Prediction Dataset.csv")

st.title("פרויקט 1 למידת מכונה- נדלן")
st.header("נוצר על ידי יהלי סביון")

# הצגת הטבלה באתר
st.dataframe(df.head())

# 2. הגדרת משתנה המטרה והעמודות לחיזוי
y = df['Price']
X = df[['Area', 'Location', 'Condition', 'Bedrooms']]

# 3. יצירת שלושה גרפים להצגת הנתונים
# שומרים את הגרף לתוך המשתנה fig
fig = plt.figure(figsize=(18, 5))

# גרף 1: מחירו הממוצע של בית לפי אזור 
plt.subplot(1, 3, 1)
sns.barplot(data=df, x='Location', y='Price', errorbar=None)
plt.title('Relationship between Location and Average Price')
plt.xlabel('Location')
plt.ylabel('Average price ($)')

# גרף 2: מחיר ממוצע לפי מצב הנכס
plt.subplot(1, 3, 2)
sns.barplot(data=df, x='Condition', y='Price', errorbar=None, order=['Poor', 'Fair', 'Good', 'Excellent'])
plt.title('Relationship between Condition and Average Price')
plt.xlabel('Condition')
plt.ylabel('Average price ($)')

# גרף 3: הקשר בין שטח הבית למחיר (פיזור נתונים)
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Area', y='Price', alpha=0.6)
plt.title('Relationship between Area and Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')

plt.tight_layout()

# פקודת ההצגה של הגרף ב-Streamlit
st.pyplot(fig)
