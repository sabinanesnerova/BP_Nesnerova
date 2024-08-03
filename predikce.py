import numpy as np
import matplotlib.pyplot as plt
import definice_funkci as df
import linearni_regrese_bak as lrb
import statsmodels.api as sm
X = lrb.X
y = lrb.y
X_bez1 = X [:,1] #vymaže první sloupec (jedničky) z matice
nove_x = np.array([[1.0, float(input("Zadejte výšku pro kterou chcete predikovat hmotnost: \n "))]]) 
nove_x_bez1 = nove_x[:, 1] #vymaže první hodnotu z nového řádku matice
#predikce pomocí definovaných funkcí:
odhadovane_y = df.odhad_zavisle(nove_x, lrb.b) #program spočítá odhadované y na základě rovnice pro lineární regresi
odchylka_y_spol = df.odchylka_odhadu_int_spol(X, y, nove_x, lrb.b) #výpočet odchylky pro odhad
interval_spol = df.interval_spolehlivosti(X, y, nove_x, lrb.b) #výpočet mezí intervalu
odchylka_y_pred = df.odchylka_odhadu_int_pred(X, y, nove_x, lrb.b) 
interval_pred = df.interval_predikce(X, y, nove_x, lrb.b) 
#predikce pomocí knihovny:
x_novy = np.array([[1.0, nove_x_bez1[0]]])
x_const = sm.add_constant(lrb.x)
model = sm.OLS(y, x_const)
vysledky = model.fit()
y_pred = vysledky.predict(x_novy)
pred_int = vysledky.get_prediction(x_novy).summary_frame(alpha=0.05)
dolni_meze_y = pred_int['mean_ci_lower']
horni_meze_y = pred_int['mean_ci_upper']
konst = df.intervaly_koeficientu(X, y, lrb.b)[0]
koef = df.intervaly_koeficientu(X, y, lrb.b)[1]
############################################
#                                          #
#           VÝPIS VÝSLEDKŮ                 #
#                                          #
############################################
print(f"odhadované y pro x={nove_x_bez1} je: {odhadovane_y}\n")
print(f"odhad y s odchylkou je: {odhadovane_y}±{odchylka_y_spol}\n")
print(f" y se pohybuje v tomto intervalu: ({interval_spol})\n")
print(f"Interval predikce je: ({interval_pred}) ")
print("----------------------------------------------")
print("Pomocí knihovny:")
print(f"Predikované y: {y_pred[0]}")
print(f"Interval spolehlivosti: ({dolni_meze_y[0]}, {horni_meze_y[0]})")
############################################
#                                          #
#                  GRAF                    #
#                                          #
############################################
leve_meze_list=[]
prave_meze_list=[]
leve_meze_pred_list=[]
prave_meze_pred_list=[]

for j in X:
    interval_spolehlivosti = df.interval_spolehlivosti(X, y, j, lrb.b)
    interval_predikce = df.interval_predikce(X, y, j, lrb.b)
    leve_meze = interval_spolehlivosti[0]
    prave_meze = interval_spolehlivosti[1]
    leve_meze_pred = interval_predikce[0]
    prave_meze_pred = interval_predikce[1]
    leve_meze_list.append(leve_meze)
    prave_meze_list.append(prave_meze)
    leve_meze_pred_list.append(leve_meze_pred)
    prave_meze_pred_list.append(prave_meze_pred)

plt.plot(lrb.x, lrb.konstanta + lrb.koeficient_x*lrb.x, color='lightblue', label='Lineární regrese')
plt.scatter(X_bez1, y, color='purple', label='Data', marker = 'x', zorder = 5 ) # vykreslení dat
plt.xlabel("Výška [m]", fontsize=10)
plt.ylabel("Hmotnost [kg]", fontsize=10)
plt.plot(X_bez1, prave_meze_list, color='red', label='Pásy spolehlivosti', linestyle='--') 
plt.plot(X_bez1, leve_meze_list, color='red',  linestyle='--') 
plt.plot(X_bez1, prave_meze_pred_list, color='green', label='Pásy predikce', linestyle='--') #vykreslení horních intervalů jakožto přímky
plt.plot(X_bez1, leve_meze_pred_list, color='green',  linestyle='--') #vykreslení dolních intervalů jakožto přímky
plt.vlines(nove_x_bez1, ymin=interval_pred[0], ymax=interval_pred[1], color='darkblue', linestyle='-', zorder = 5)
plt.hlines(y=interval_pred[0], xmin=nove_x_bez1-0.005, xmax=nove_x_bez1+0.005, colors='darkblue', linestyles='-')
plt.hlines(y=interval_pred[1], xmin=nove_x_bez1-0.005, xmax=nove_x_bez1+0.005, colors='darkblue', linestyles='-')
plt.scatter(nove_x_bez1, odhadovane_y, color = 'black', marker = 's', label = 'Predikce pro zadané x', zorder=5) #bod predikce
plt.legend(fontsize=10)
plt.xlim(1.46, 1.84)
plt.ylim(45, 80)
plt.legend(fontsize=10)
plt.legend(fontsize=10)
plt.show()


