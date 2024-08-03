import matplotlib.pyplot as plt
import numpy as np
import definice_funkci as dfu
from scipy.stats import f
x_1 = [1,1.2, 1.5, 1.8, 1.9, 2.1,1.7,1.9,2 ,1.4,1.6,1.1, 1.1]
y_1 = [10,12,14,16,16,18,15,17,17,13,15,12,11]
X_1 = np.column_stack([np.ones(len(x_1)), x_1])  # Vloží sloupec 1 do matice pro konstantu
b = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y_1  # @ je maticové násobení, T je transpozice
konstanta_1 = b[0]
koeficient_x_1 = b[1]
print("Koeficient determinace",dfu.regresni_koeficient(X_1, y_1, b))
print(dfu.outlier(X_1, y_1, b))
print(dfu.leverage_point(X_1))
cook_vzdalenosti = dfu.cook_vzd(X_1, y_1, b)
influent_nalezen = False
for k, cook_vzd in enumerate(cook_vzdalenosti):
    if cook_vzd > f.ppf(0.5, len(b), len(y_1)-len(b)):
        print(f"bod {X_1[k]}, {y_1[k]} je influentní")
        influent_nalezen = True
    if not influent_nalezen:
        print("influent nenalezen") 
print("---------------------")
#outlier
x_2 = x_1 + [1.7]
y_2 = y_1 + [20]
X_2 = np.column_stack([np.ones(len(x_2)), x_2])  
b = np.linalg.inv(X_2.T @ X_2) @ X_2.T @ y_2  
konstanta = b[0]
koeficient_x = b[1]
print(dfu.regresni_koeficient(X_2, y_2, b))
z = np.linspace(1,2.1,100)
plt.scatter(x_1, y_1, zorder = 5, color = "#9370DB", label = "Data")
plt.plot(x_2, konstanta + koeficient_x*np.array(x_2))
plt.scatter(1.7,20, zorder = 5, marker = 's', label = "Outlier")
plt.xlim(0.9, 2.8) 
plt.ylim(7.5,22.5) 
print(dfu.outlier(X_2, y_2, b))
print(dfu.leverage_point(X_2))
cook_vzdalenosti_2 = dfu.cook_vzd(X_2, y_2, b)
influent_nalezen_2 = False
for k, cook_vzd in enumerate(cook_vzdalenosti_2):
    if cook_vzd > f.ppf(0.5, len(b), len(y_2)-len(b)):
        print(f"bod {X_2[k]}, {y_2[k]} je influentní")
        influent_nalezen_2 = True
    if not influent_nalezen_2:
        print("influent nenalezen")
print("---------------------")
#influent
x_3 = x_1 + [2]
y_3 = y_1 + [8]

X_3 = np.column_stack([np.ones(len(x_3)), x_3])
b = np.linalg.inv(X_3.T @ X_3) @ X_3.T @ y_3  
konstanta_3 = b[0]
koeficient_x_3 = b[1]
z_3 = np.linspace(1,2.5,100)
print(dfu.outlier(X_3, y_3, b))
print(dfu.leverage_point(X_3))
print(dfu.regresni_koeficient(X_3, y_3, b))
cook_vzdalenosti_3 = dfu.cook_vzd(X_3, y_3, b)
influent_nalezen_3 = False
for k, cook_vzd in enumerate(cook_vzdalenosti_3):
    if cook_vzd > f.ppf(0.5, len(b), len(y_3)-len(b)):
        print(f"bod {X_3[k]}, {y_3[k]} je influentní")
        influent_nalezen_3 = True
    if not influent_nalezen_3:
        print("influent nenalezen") 
print("---------------------")
plt.scatter(2, 8, color = "darkblue", zorder = 5, marker = 's', label = "Influentní bod")
x_4 = x_1 + [2.75]
y_4 = y_1 + [22]
X_4 = np.column_stack([np.ones(len(x_4)), x_4]) 
b = np.linalg.inv(X_4.T @ X_4) @ X_4.T @ y_4  
konstanta_4 = b[0]
koeficient_x_4 = b[1]
z_4 = np.linspace(1, 3, 100)
print(dfu.regresni_koeficient(X_4, y_4, b))
print(dfu.outlier(X_4, y_4, b))
print(dfu.leverage_point(X_4))
cook_vzdalenosti_4 = dfu.cook_vzd(X_4, y_4, b)
influent_nalezen_4 = False
for k, cook_vzd in enumerate(cook_vzdalenosti_4):
    if cook_vzd > f.ppf(0.5, len(b), len(y_4)-len(b)):
        print(f"bod {X_4[k]}, {y_4[k]} je influentní")
        influent_nalezen_4 = True
    if not influent_nalezen_4:
        print("influent nenalezen") 
plt.scatter(2.75,22, color = "lightblue",  zorder = 5, marker = 's', label = "Leverage bod")
plt.plot(x_4, konstanta_4 + koeficient_x_4*np.array(x_4), color ='lightblue')
plt.plot(x_3, konstanta_3 + koeficient_x_3*np.array(x_3), color = 'darkblue')
plt.plot(z, konstanta_1 + koeficient_x_1*np.array(z), linestyle='--', linewidth=1.3, dashes=(5, 5), label = "Regrese bez outlierů", color = "purple") 
plt.legend(fontsize=10, loc='lower right', bbox_to_anchor=(1, 0.05))
plt.show()




