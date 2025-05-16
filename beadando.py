'''
A mellékelt adathalmaz a különböző gyorséttermekben kapható ételek kalória- és egyéb adatait tartalmazza.

Felelősségek lebontása:

Adatfeldolgozás
Modellezés
Prototipizálás / teszelés
Főbb megválaszolandó kérdések:

Hányféle étteremlánc adatai találhatóak meg az adathalmazban?
Éttermenként átlagosan hány termék található az adathalmazban? Mi ennek a mediánja, szórása, mik a szélsőértékei? (milyen tartományban mozog...)
Melyik étteremben használják a legtöbb sót, legtöbb cukrot?
Melyik éttermet válasszuk, ha maximalizálni szeretnénk a fehérje bevitelt, a lehető legkevesebb kalória mellett?
Előrejelezhető a különböző adatok (zsír, cukor, só, rostok, fehérje, szénhidrátok...) segítségével a kalóriatartalom?
'''

# 1. Alap importok
import numpy as np
import pandas as pd

# 2. Felügyelt tanulás k-legküzelebbi szomszéd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# 3. Train Test Splits, Cross Validation és Lineáris Regresszion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# 4. Kluszterezési módszerek
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


# Adatfeldolgozás
# Adatok beolvasása
df = pd.read_csv('fastfood_calories.csv')
# Adatok megtekintése
print(df.head())
# Adatok típusainak ellenőrzése
print(df.dtypes)
# Adatok hiányosságainak ellenőrzése
print(df.isnull().sum())
# Adatok tisztítása
df = df.dropna()  # Hiányzó értékek eltávolítása


# 1. Hányféle étteremlánc adatai találhatóak meg az adathalmazban?
print("1__________________________________")
# Az étteremláncok neveinek egyedi értékeinek száma
unique_chains = df['restaurant'].nunique()
print(f'Hányféle étteremlánc adatai találhatóak meg az adathalmazban? {unique_chains}')

# 2. Éttermenként átlagosan hány termék található az adathalmazban? Mi ennek a mediánja, szórása, mik a szélsőértékei? (milyen tartományban mozog...)
print("2__________________________________")
# Számoljuk meg éttermenként a termékek számát
products_per_restaurant = df.groupby('restaurant').size()

# Átlag termékszám éttermenként
mean_products = products_per_restaurant.mean()
# Medián termékszám éttermenként
median_products = products_per_restaurant.median()
# Szórás a termékszámokban
std_products = products_per_restaurant.std()
# Minimum és maximum termékszám
min_products = products_per_restaurant.min()
max_products = products_per_restaurant.max()
# Tartomány (range)
range_products = max_products - min_products

print("\nÉttermenként található termékek statisztikái:")
print(f"Átlagos termékszám: {mean_products:.2f}")
print(f"Medián termékszám: {median_products:.0f}")
print(f"Termékszám szórása: {std_products:.2f}")
print(f"Minimum termékszám: {min_products} ({products_per_restaurant.idxmin()})")
print(f"Maximum termékszám: {max_products} ({products_per_restaurant.idxmax()})")
print(f"Termékszám tartománya: {range_products}")

# A termékek eloszlásának vizualizálása
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
products_per_restaurant.sort_values().plot(kind='bar')
plt.title('Termékek száma éttermenként')
plt.xlabel('Étterem')
plt.ylabel('Termékek száma')
plt.tight_layout()
plt.savefig('termekek_szama_ettermenként.png')
plt.close()

# Részletes lista az éttermekről és termékszámokról
print("\nÉttermenként található termékek száma:")
for restaurant, count in products_per_restaurant.sort_values(ascending=False).items():
    print(f"{restaurant}: {count} termék")

# 3. Melyik étteremben használják a legtöbb sót, legtöbb cukrot?
print("3__________________________________")
# A só és cukor mennyiségének összegzése étteremlánc szerint
sodium_per_restaurant = df.groupby('restaurant')['sodium'].sum()
sugar_per_restaurant = df.groupby('restaurant')['sugar'].sum()
# Legtöbb sót használó étterem
max_sodium_restaurant = sodium_per_restaurant.idxmax()
max_sodium_value = sodium_per_restaurant.max()
# Legtöbb cukrot használó étterem
max_sugar_restaurant = sugar_per_restaurant.idxmax()
max_sugar_value = sugar_per_restaurant.max()
print(f'\nLegtöbb sót használó étterem: {max_sodium_restaurant} ({max_sodium_value} mg)')
print(f'Legtöbb cukrot használó étterem: {max_sugar_restaurant} ({max_sugar_value} g)')
# A só és cukor mennyiségének vizualizálása
plt.figure(figsize=(12, 6))
sodium_per_restaurant.sort_values().plot(kind='bar', color='blue', label='Só (mg)')
sugar_per_restaurant.sort_values().plot(kind='bar', color='red', label='Cukor (g)', alpha=0.7)
plt.title('Só és cukor mennyisége étteremlánc szerint')
plt.xlabel('Étteremlánc')
plt.ylabel('Mennyiség')
plt.legend()
plt.tight_layout()
plt.savefig('só_cukor_mennyiseg.png')
plt.close()

# 4. Melyik éttermet válasszuk, ha maximalizálni szeretnénk a fehérje bevitelt, a lehető legkevesebb kalória mellett?
print("4__________________________________")

# Fehérje-kalória arány kiszámítása minden termékre
# Magasabb érték = több fehérje kevesebb kalóriával
df['protein_per_calorie'] = df['protein'] / df['calories']
# Átlagos fehérje-kalória arány éttermenként
protein_efficiency = df.groupby('restaurant')['protein_per_calorie'].mean()

# Rendezzük csökkenő sorrendbe
protein_efficiency_sorted = protein_efficiency.sort_values(ascending=False)
print("\nÉttermek fehérje-kalória aránya (fehérje/kalória):")
for restaurant, ratio in protein_efficiency_sorted.items():
    print(f"{restaurant}: {ratio:.4f}")

# Legjobb étterem kiválasztása
best_restaurant = protein_efficiency_sorted.index[0]
best_ratio = protein_efficiency_sorted.iloc[0]
print(f"\nLegjobb fehérje-kalória arányú étterem: {best_restaurant} ({best_ratio:.4f})")

# Eredmény ábrázolása
plt.figure(figsize=(12, 6))
protein_efficiency_sorted.plot(kind='bar', color='green')
plt.title('Éttermek fehérje-kalória aránya')
plt.xlabel('Étterem')
plt.ylabel('Fehérje/kalória arány')
plt.axhline(y=protein_efficiency.mean(), color='red', linestyle='--', label='Átlag')
plt.legend(['Átlagos fehérje-kalória arány'])
plt.tight_layout()
plt.savefig('feherje_kaloria_arany.png')
plt.close()

#--- Érdekelt minket úgyhogy: részletesebb vizsgálat: a legmagasabb fehérje-kalória arányú termékek az adatbázisban
top_items = df.sort_values('protein_per_calorie', ascending=False).head(10)
print("\nLegjobb fehérje-kalória arányú termékek:")
for idx, row in top_items.iterrows():
    print(f"{row['restaurant']} - {row['item']}: {row['protein_per_calorie']:.4f} (Fehérje: {row['protein']}g, Kalória: {row['calories']})")
#---

# 5. Előrejelezhető a különböző adatok (zsír, cukor, só, rostok, fehérje, szénhidrátok...) segítségével a kalóriatartalom?
print("5__________________________________")

# Kiválasztjuk a prediktorokat (magyarázó változókat) és a célt (kalória)

# Mivel salad-ból mindegyik "Other" ezért nem használjuk fel
predictors = ['total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium', 
              'total_carb', 'fiber', 'sugar', 'protein', 'vit_a', 'vit_c', 'calcium']
X = df[predictors]
y = df['calories']

# Felosztjuk az adatokat tanuló és teszt halmazokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Lineáris regressziós modellt hozunk létre és illesztjük a tanító adatokra
model = LinearRegression()
model.fit(X_train, y_train)

# Előrejelzés a teszt adatokon
y_pred = model.predict(X_test)

# Modell teljesítményének értékelése
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

print("\nLineáris regressziós modell teljesítménye:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} kalória")
print(f"R² (Determinációs együttható): {r2:.4f}")

# Vizsgáljuk meg a prediktorok fontosságát (együtthatók)
importance = pd.DataFrame({
    'Prediktor': predictors,
    'Együttható': model.coef_
})
importance['Abszolút_együttható'] = np.abs(importance['Együttható'])
importance = importance.sort_values('Abszolút_együttható', ascending=False)

print("\nPrediktorok fontossága (együtthatók abszolút értéke szerint rendezve):")
for index, row in importance.iterrows():
    print(f"{row['Prediktor']}: {row['Együttható']:.4f}")

# Ábrázoljuk a tényleges vs. előrejelzett értékeket
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Tényleges kalória')
plt.ylabel('Előrejelzett kalória')
plt.title('Tényleges vs. Előrejelzett kalóriaértékek')
plt.tight_layout()
plt.savefig('kaloria_elorejelzes.png')
plt.close()

# Ábrázoljuk a prediktorok fontosságát
plt.figure(figsize=(12, 6))
importance.plot(kind='bar', x='Prediktor', y='Abszolút_együttható', color='teal')
plt.title('Prediktorok fontossága a kalóriatartalom előrejelzésében')
plt.xlabel('Tápanyagok')
plt.ylabel('Együtthatók abszolút értéke')
plt.tight_layout()
plt.savefig('prediktor_fontossag.png')
plt.close()

# A kapott eredmények értelmezése:
# 
# A lineáris regressziós modell kiváló teljesítményt mutat a kalóriatartalom előrejelzésében:
# - RMSE = 21.42 kalória: A modell átlagosan csak ~21 kalóriát téved, ami nagyon pontos
# - R² = 0.9924: A modell a kalóriatartalom varianciájának 99.24%-át magyarázza meg
# 
# A legfontosabb prediktorok (együtthatók szerint):
# 1. Transzzsír (12.16): Legerősebb hatás, 1g transzzsír ~12 kalóriát jelent
# 2. Összes zsír (7.09): Jelentős pozitív hatás
# 3. Összes szénhidrát (3.92): Közepes pozitív hatás
# 4. Fehérje (3.87): Hasonló mértékű befolyás, mint a szénhidrátoknál
# 5. Rost (2.85): Meglepően erős pozitív együttható
# 
# A nátriumnak (0.0061) gyakorlatilag nincs hatása a kalóriatartalomra, ami érthető,
# mivel a só nem tartalmaz kalóriát. Néhány összetevő (cukor, vitaminok, kalcium) 
# negatív együtthatóval rendelkezik, ami vagy a modelltől, vagy az ételek 
# összetételének sajátosságaiból adódhat.