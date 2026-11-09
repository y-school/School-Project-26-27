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

st.subheader("על הדאטה")
st.write("הדאטה עוסק במחירי בתים ובתכונות השונות שלהם. כל שורה בטבלה מייצגת בית אחד ספציפי, וכוללת מידע עליו כמו שטח, מצב פיזי, מיקום, מספר חדרי שינה והמחיר שלו.")
st.markdown("**ניתן לראות שיש 10 עמודות, כל אחת מייצגת דבר שונה:**")
st.write("Id- תעודת הזהות של הבית (עמודת מספר)")
st.write("Area - שטח הבית (עמודת מספר)")
st.write("Bedrooms- מספר החדרים בבית (עמודה מספר)")
st.write("Bathrooms- מספר השירותים בבית (עמודה מספר)")
st.write("Floors- מספר הקומות בבית (עמודה מספר)")
st.write("YearBuilt- שנת הבנייה של הבית (עמודה מספר)")
st.write("Location- מיקום הבית (עמודת קטגוריה)")
st.write("Condition- מצב הבית (עמודת קטגוריה)")
st.write("Garage- האם יש חניה (עמודת קטגוריה)")
st.write("Price- מחיר הבית (עמודת מספר)")

st.subheader("מידע על הנתונים")
st.write("בטבלה יש 2000 שורות ו-10 עמודות.")

st.write("בטבלה אין ערכים חסרים")

st.write("ממוצעים ונתונים סטטיסטיים (כמו מינימום ומקסימום):")
# הפונקציה describe מציגה אוטומטית ממוצעים, סטיות תקן ועוד
st.dataframe(df.describe())

st.subheader("מה נרצה לחזות (y) ובאמצעות אילו עמודות (X)")
st.write("**משתנה המטרה (y):** המחיר (Price) - זהו הערך שאנחנו רוצים שהמודל שלנו ילמד לחזות.")
st.write("**עמודות לחיזוי (X):** שטח הבית (Area), מיקום (Location), מצב הנכס (Condition) ומספר חדרי השינה (Bedrooms).")
st.write("**נימוק לבחירה:** אלו המאפיינים המרכזיים ביותר שמשפיעים על שווי של נדל״ן במציאות. גודל הבית ומספר החדרים מכתיבים את המרחב הבסיסי, המיקום משפיע על הביקוש ועל יוקרת האזור, ומצב הנכס קובע את איכות המגורים בו. שילוב של ארבעתם ייתן למודל בסיס חזק להערכת המחיר.")

# 2. הגדרת משתנה המטרה והעמודות לחיזוי
y = df['Price']
X = df[['Area', 'Location', 'Condition', 'Bedrooms']]

# 3. יצירת שלושה גרפים להצגת הנתונים
# שומרים את הגרף לתוך המשתנה fig
fig = plt.figure(figsize=(14, 10))

# גרף 1: מחירו הממוצע של בית לפי אזור 
plt.subplot(2, 2, 1)
sns.barplot(data=df, x='Location', y='Price', errorbar=None)
plt.title('Relationship between House Location and Average Price')
plt.xlabel('Location')
plt.ylabel('Average price ($)')

# גרף 2: מחיר ממוצע לפי מצב הנכס
plt.subplot(2, 2, 2)
sns.countplot(data=df, x='Location')
plt.title('Number of houses by location')
plt.xlabel('Location')
plt.ylabel('Number of houses')

# גרף 3: הקשר בין שטח הבית למחיר (פיזור נתונים)
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='Price', kde=False)
plt.title('Distribution of House Prices')
plt.xlabel('Price ($)')
plt.ylabel('Number of houses')
plt.ticklabel_format(style='plain', axis='x')

plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Area', y='Price')
plt.title('Relationship between House Area and Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()

# פקודת ההצגה של הגרף ב-Streamlit
st.pyplot(fig)
