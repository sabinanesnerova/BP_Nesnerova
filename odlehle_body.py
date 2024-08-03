import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import definice_funkci as df
from scipy.stats import f
data = pd.read_excel("zavislost vahy na vysce.xlsx")
data = data.iloc[:, :2]
data.describe()[["Výška", "Váha" ]]
x = data["Výška"].to_numpy()
y = data["Váha"].to_numpy() 
x = x.reshape(-1,1)
y = y.reshape(-1,1)
#nalezení koeficientů:
X = np.column_stack([np.ones(len(x)), x])  
b = np.linalg.inv(X.T @ X) @ X.T @ y
konstanta_bez = b[0]
koeficient_x_bez = b[1]
print(df.regresni_koeficient(X,y, b))
############################################
#                                          #
#                 GRAF                     #
#                                          #
############################################
plt.plot(x, konstanta_bez+koeficient_x_bez*x, color = 'red', label = "$y = \\beta_0 + x \\beta_1$") #vykreslení dat
plt.scatter(x,y, label="Data") #vykreslení dat, která jsou k dispozici
plt.xlabel("Výška [m]", fontsize = 10)
plt.ylabel("Hmotnost [kg]", fontsize = 10)
plt.legend(fontsize = 10)
plt.show()
############################################
#                                          #
#       DATA BEZ ODLEHLÉHO BODU            #
#                                          #
############################################
x_outlier = x[df.outlier(X,y,b)] #označení outliera
y_outlier = y[df.outlier(X,y,b)]
for x_val, y_val in zip(x_outlier, y_outlier):
    print(f"Outlier je [{x_val};{y_val}]\n")  #vypsání nalezeného outliera
if x_outlier.any() and y_outlier.any(): #pokud nějaký nalezen -> vymazání
    k_vymazani = np.where(df.outlier(X,y,b))[0]
    x = np.delete(x, k_vymazani)
    y = np.delete(y, k_vymazani)
else: print("Outlier nenalezen\n")
############################################
#                                          #
#           DATA S ODLEHLÝM BODEM          #
#                                          #
############################################
data.loc[len(data.index)] = [1.5, 72] #uměle přidán outlier
x = data["Výška"].to_numpy()
y = data["Váha"].to_numpy()
#nalezení koeficientů pro data s outlierem:
X = np.column_stack([np.ones(len(x)), x])  #vloží sloupec 1 do matice pro konstantu
b = np.linalg.inv(X.T @ X) @ X.T @ y # @ je maticové násobení, T je transpozice
konstanta = b[0]
koeficient_x = b[1]
print("Data s outlierem:")
print(data)
print(df.regresni_koeficient(X, y, b))
x_s = x 
#stejný proces jako v horních řádcích:
x_outlier = x[df.outlier(X,y,b)]
y_outlier = y[df.outlier(X,y,b)]

for x_out, y_out in zip(x_outlier, y_outlier):
    print(f"Outlier je [{x_out};{y_out}]\n")  
if x_outlier.any() and y_outlier.any():
    k_vymazani = np.where(df.outlier(X, y, b))[0]
    data = data.drop(k_vymazani)
    x_bez = np.delete(x, k_vymazani)
    y_bez = np.delete(y, k_vymazani)
    print("Outlier nalezen a odstraněn")
else: print("Outlier nenalezen")

lev_pravda = np.where(df.leverage_point(X)==True)
x_lev = np.array(x)[lev_pravda]
y_lev = np.array(y)[lev_pravda]
for nalez in df.leverage_point(X):
   if nalez ==True:
       print(f"\nv datech se vyskytuje leverage point a je to: {x_lev}, {y_lev}")        
   else: print("\nLeverage nenalezen")
influent_nalezen = False   
cook_vzdalenosti = df.cook_vzd(X, y, b)
for i, cook_vzd in enumerate(cook_vzdalenosti):
       if cook_vzd > f.ppf(0.5, len(b), len(y)-len(b)):
            print(f"bod ({x[i]},{y[i]}) je influentní")
            influent_nalezen = True
       if not influent_nalezen: print("influent nenalezen") 
print("Nová data:") #data po vymazání outliera
print(data)       
############################################
#                                          #
#                 GRAF                     #
#                                          #
############################################
plt.plot(x, konstanta_bez+koeficient_x_bez*x, color = 'lightblue', label = "Lineární regrese bez outliera", linestyle = "--") #vykreslení dat
plt.plot(x_s, konstanta+koeficient_x*x_s, color = 'green', label = "Lineární regrese s outlierem") #vykreslení dat
plt.scatter(x,y, label = "Data s outlierem", c="darkblue", zorder=5) #data obsahující outlier
plt.xlabel("Výška [m]", fontsize = 10)
plt.ylabel("Hmotnost [kg]", fontsize = 10)
plt.scatter(x_outlier, y_outlier, marker = '*', c="red", label='Outlier') 
plt.legend(fontsize = 10)
plt.show()

