import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import definice_funkci as df
import statsmodels.stats.api as sms
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import f
from scipy.stats import t
data = pd.read_excel("zavislost vahy na vysce.xlsx")
data.describe()[["Váha", "Výška" ]]
x = data["Výška"].to_numpy()
y = data["Váha"].to_numpy() 
x = x.reshape(-1,1)
y = y.reshape(-1,1)
#řešení rovnic:
X = np.column_stack([np.ones(len(x)), x])  #vloží sloupec 1 do matice pro konstantu
b = np.linalg.inv(X.T @ X) @ X.T @ y 
konstanta = b[0]
koeficient_x = b[1]
#ověření předpokladů:
rezidua = df.reziduum(X,y,b)
shapiro_test, p_value = stats.shapiro(rezidua)
dw_statistika = sms.durbin_watson(rezidua)
skupina1 = rezidua[:6]  
skupina2 = rezidua[6:12] 
skupina3 = rezidua[12:] 
f_statistic, pvalue = stats.f_oneway(skupina1, skupina2, skupina3) #na základě rozdělemí dat do tří skupin
#přidání outliera a vyšetření
novy_bod = pd.DataFrame({"Výška": [1.50], "Váha": [72]})
data = pd.concat([data, novy_bod], ignore_index=True)
x_n = data["Výška"].to_numpy()
y_n = data["Váha"].to_numpy() 
x_n = x_n.reshape(-1,1)
y_n = y_n.reshape(-1,1)
X_n = np.column_stack([np.ones(len(x_n)), x_n]) 
b_n = np.linalg.inv(X_n.T @ X_n) @ X_n.T @ y_n
outlier = df.outlier(X_n, y_n, b_n)
leverage = df.leverage_point(X_n)
cook_vzdalenosti = df.cook_vzd(X_n, y_n, b_n)
leverage_point = []
influentni_bod = []
outlier_ = []
for k, cook_vzd in enumerate(cook_vzdalenosti):
       if cook_vzd > f.ppf(0.5, len(b), len(y_n)-len(b)):
            influentni_bod.append(f"bod {x_n[k]}, {y_n[k]} je influentní")
       else: influentni_bod.append("Influentní bod nebyl nalezen")
for k, lev in enumerate(leverage):
       if lev ==True:
            leverage_point.append(f"bod {x_n[k]}, {y_n[k]} je leverage")
       else: leverage_point.append("leverage nenalezen") 
for k, out in enumerate(outlier):
       if out ==True:
            outlier_.append(f"bod {x_n[k]}, {y_n[k]} je outlier")
       else: outlier_.append("Outlier nebyl nalezen")
############################################
#                                          #
#               SKRIPT                     #
#                                          #
############################################
#výpis výsledků lineárního modelu
print('Shapiro-Wilk test:', shapiro_test )
print('Durbin-Watson:', dw_statistika)
print('F-statistika:', f_statistic)
for lev_point, out_, infl_bod in zip(leverage_point, outlier_, influentni_bod):
   print(f"{lev_point}, {out_}, {infl_bod} \n")
print(f"řešení regrese pomocí rovnic: y = {konstanta} + {koeficient_x}x \n" ) #vypíše rovnici mého modelu
print(f"regresní koeficient je {df.regresni_koeficient(X, y, b)}\n") #vypíše regresní koeficient pro tento model -> ukazuje kvalitu
print(f'Pro {len(x)-2} stupňů volnosti je q995 = {t.ppf(1-(0.05/2), len(y)-len(b))} \n') #vypíše kriticku hodnotu studentova t-testu pro 3 stupně volnosti
print(f"T pro odhad \u03B2\u2080 je: {df.t_test_koeficient(X,y,b)[0]}\n")
if (abs(df.t_test_koeficient(X,y,b)[0]) >= df.hodnota_student_rozdeleni(len(y)-len(b))): #ověření hypotézy, že beta0 je nula, pomocí studentova t-testu
    print("hypotéza \u03B2\u2080 = 0 neplatí\n")
else:print("\u03B2\u2080 = 0\n")
print(f"T pro odhad \u03B2\u2081 je: {df.t_test_koeficient(X,y,b)[1]}\n") 
if (abs(df.t_test_koeficient(X,y,b)[1]) >= df.hodnota_student_rozdeleni(len(y)-len(b))): #ověření hypotézy, že beta1 je nula, pomocí studentova t-testu
    print("hypotéza \u03B2\u2081 = 0 neplatí\n")
else: print("\u03B2\u2081 = 0\n")
print(f"F-statistika pro celkový F-test je: {df.F_test(X, y, b)}")
if df.F_test(X, y, b)> f.ppf(0.95, len(b)-1, len(y)-len(b)):
    print("Hypotéza o nulovosti všech koeficientů neplatí")
else: print("Všechny koeficienty jsou nulové")
#vypíše intervaly, ve kterých se s 95% pravděpodobností vyskytuje beta0 a beta1 na základě jednotlivých odhadů
print(f"interval spolehlivosti pro \u03B2\u2080 a \u03B2\u2081 je {df.intervaly_koeficientu(X, y, b)}\n")
print(f"p-hodnoty pro jednotlivé koeficienty jsou {df.p_hodnota_koeficient_i(X, y, b)}")
############################################
#                                          #
#               SKRIPT                     #
#                                          #
############################################
#lineární regrese pomocí knihovny:
reg = LinearRegression() #určení modelu
reg.fit(x,y)
konstanta_knih = reg.intercept_
koeficient_x_knih = reg.coef_ 
x_const = sm.add_constant(x)
model = sm.OLS(y, x_const)
vysledky = model.fit()
print(vysledky.summary())
print("Lineární regrese pomocí knihovny LinearRegression.")
print(reg.score(x,y)) #koeficient regrese -> kvalita modelu
print(f"\u03B2\u2080 je: {konstanta_knih}")
print(f"\u03B2\u2081 je: {koeficient_x}")
print(f" \n rovnice vypočtená pomocí knihovny: y = {konstanta_knih} + {koeficient_x_knih}x \n") # vypíše rovnici lineární regrese
############################################
#                                          #
#                 GRAFY                     #
#                                          #
############################################
#regrese s outlierem
plt.scatter(x,y, c="darkblue", label = "Data", zorder = 5 )
plt.plot(x, konstanta + koeficient_x*x, color='lightblue', label='Lineární regrese', linestyle = '--')
plt.scatter(1.5,72, c="red", label = "Outlier", marker = '*', zorder = 5 )
plt.plot(x_n, b_n[0] + b_n[1]*x_n, color='green', label='Lineární regrese s outlierem')
#plt.title("Závislost rychlosti na teplotě")
plt.xlabel("Výška [m]", fontsize=10)
plt.ylabel("Hmotnost [kg]", fontsize=10)
plt.legend(fontsize=10)
plt.show()
#ilustrace předpokladu
odhady = konstanta + koeficient_x*x
k = np.zeros_like(x)
plt.plot(odhady, k, color = 'darkblue')
plt.scatter(odhady, df.studentizovane_reziduum(X, y, b))
plt.xlabel("Odhady Y", fontsize = 10)
plt.ylabel("Studentizovaná rezidua", fontsize=10)
plt.show()


