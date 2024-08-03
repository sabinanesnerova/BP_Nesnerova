import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import f
import definice_funkci as dfu
import matplotlib.pyplot as plt
from scipy.stats import chi2
df = pd.read_excel("data_git.xlsx", sheet_name="postcranial final_corr") 
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1})
data = df[['sex','f12']]
data.dropna(subset=['sex', 'f12'], how='any', inplace=True)  # odstranění řádků s chybějícími hodnotami
y = data['sex'].values
X = data[['f12']].values  
nezavisle = data['f12']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #rozdělení na trénovací a testovací
koeficienty, shrnuti = dfu.model_logit(y_train,X_train)
X_train_pred = np.column_stack([np.ones(len(X_train)), X_train]) #přidání jednotkového sloupce pro konstantu
#identifikace odlehlých bodů
outlier = dfu.outlier_log(X_train_pred, y_train, koeficienty)
leverage = dfu.leverage_point_log(X_train_pred, koeficienty)
cook_vzdalenosti = dfu.cook_vzd_log(X_train_pred, y_train, koeficienty)
influent_nalezen = False
outlier_nalezen = False
leverage_nalezen = False
for i, cook_vzd in enumerate(cook_vzdalenosti):
       if cook_vzd > f.ppf(0.5, len(koeficienty), len(y_train)-len(koeficienty)):
            print(f"bod {X_train_pred[i]} je influentní")
            influent_nalezen = True
if not influent_nalezen:
    print("influent nenalezen") 

for i, lev in enumerate(leverage):
       if lev ==True:
            print(f"bod {X_train_pred[i]} je leverage")
            leverage_nalezen = True
if not leverage_nalezen:
    print("leverage nenalezen") 

for i, out in enumerate(outlier):
       if out ==True:
            print(f"bod {X_train_pred[i]}, {y_train[i]} je outlier ")
            outlier_nalezen = True
if not outlier_nalezen:
    print("outlier nenalezen")                    
################################################## 
#                                                #
#             JEDNOTLIVÉ STATISTIKY              #
#                                                #
##################################################                                       
predikce_train =  dfu.y_model(X_train_pred, koeficienty) #předpovídá hodnoty    
binarizovane_predikce_train = (predikce_train > 0.5).astype(int) #rozdělení na 0  1
shodne_hodnoty_train = (binarizovane_predikce_train == y_train)
pocet_shodnych_zen_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 1)) #správně určené ženy
pocet_shodnych_muzu_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 0)) #správně určení muži
pocet_spatne_zen_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 1))
pocet_spatne_muzu_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 0))
presnost_train = round(dfu.presnost(pocet_shodnych_zen_train, pocet_shodnych_muzu_train, len(y_train)),2)
odchylky = np.round(dfu.odchylka_log(X_train_pred, predikce_train),2)
walduv_test = np.round(dfu.walduv_test(X_train_pred, predikce_train, koeficienty),2)      
pocet_zen = data['sex'].value_counts()[1]
pocet_muzu = data['sex'].value_counts()[0]      
X_test_pred =  np.column_stack([np.ones(len(X_test)), X_test])
predikce_test =  dfu.y_model(X_test_pred, koeficienty)
binarizovane_predikce_test = (predikce_test > 0.5).astype(int)
shodne_hodnoty_test = (binarizovane_predikce_test == y_test)
pocet_shodnych_zen_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 1))
pocet_spatne_zen_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 1))
pocet_shodnych_muzu_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 0))
pocet_spatne_muzu_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 0))
presnost_test = round(dfu.presnost(pocet_shodnych_zen_test, pocet_shodnych_muzu_test, len(y_test)),2)
nagelkerke = round(dfu.nagelkerke(X_train_pred, y_train, koeficienty),2)
f1 = round(dfu.f1_skore(pocet_shodnych_zen_test+pocet_shodnych_zen_train, pocet_spatne_zen_test+pocet_spatne_zen_train, pocet_spatne_muzu_test+pocet_spatne_muzu_train),2)
mcFadden = round(dfu.mcFadden(X_train_pred, y_train, koeficienty),2)
p_hodnoty = np.round(dfu.p_hodnota_log(X_train_pred, predikce_train, koeficienty),2)
if all(p <= 0.05 for p in p_hodnoty):
    vyznamnost = "Model je statisticky významný"
else: vyznamnost ="Nelze zamítnout nulové hypotézy"
intervaly_spol = np.round(dfu.interval_spolehlivosti_log(X_train_pred, predikce_train, koeficienty),2)
statistika = dfu.lrt(X_train_pred,y_train, koeficienty)
print(len(y_train)-len(koeficienty))
print(chi2.ppf(1 - 0.05, len(y_train)-len(koeficienty)))
if statistika > chi2.ppf(1 - 0.05, len(y_train)-len(koeficienty)):
  lrt_test = "Zamítáme nulovou hypotézu, úplný model je lepší"
else: lrt_test = "Nezamítáme nulovou hypotézu, plný model není lepší než redukovaný model."
###############################################
#                                             #
#             VÝPIS VÝSLEDKŮ                  #
#                                             #
###############################################
print("Koeficienty modelu:\n", np.round(koeficienty,2))
print("celkový počet mužů je:", pocet_muzu)
print("celkový počet žen je:", pocet_zen)
print("\nPočet správně určených žen (trénovací množina):", pocet_shodnych_zen_train)
print("\nPočet správně určených mužů (trénovací množina):", pocet_shodnych_muzu_train)
print("\nPřesnost pro sloupce (trénovací množina) je:", presnost_train)
print("Směrodatné odchylky koeficientů:", odchylky)
print("p_hodnoty pro koeficienty:", p_hodnoty)
print("Intervaly spolehlivosti pro jednotlivé koeficienty:", intervaly_spol)
print("Počet správně určených žen (testovací množina):", pocet_shodnych_zen_test)
print("\nPočet správně určených mužů (testovací množina):", pocet_shodnych_muzu_test)
print("\nPřesnost pro sloupce (testovací množina) je:", presnost_test)
print("Waldova statistika pro koeficienty je:", walduv_test)
print("F-1 skóre pro model je:", f1)
print("Nagelkerkeho koeficient je:" ,nagelkerke)
print("Mc Faddenův koeficient kvality je:", mcFadden)
print("Testy statistické významnosti jednotlivých koeficientů ukázaly, že",vyznamnost)
print("Dle LRT testu",lrt_test)
###############################################
#                                             #
#                  GRAFY                      #
#                                             #
###############################################
#křivka logistické regrese
x_hodnoty = np.linspace(X.min(), X.max(), num=100)[:, np.newaxis]
X_n = np.column_stack([np.ones(len(x_hodnoty)), x_hodnoty])
predikovane_hodnoty = dfu.y_model(X_n, koeficienty)
zena = data[data['sex'] == 1]
muz = data[data['sex'] == 0]
plt.plot(x_hodnoty, predikovane_hodnoty, c="blue", label="Logistická regrese")
plt.scatter(zena['f12'], zena['sex'], c='red', label='Žena')
plt.scatter(muz['f12'], muz['sex'], c='purple', label='Muž')
plt.xlabel("f12 [cm]", fontsize=10)
plt.ylabel("Pohlaví (0 = muž, 1 = žena)", fontsize=10)
plt.legend(fontsize=10, loc = 'center right')
plt.show()
# Histogram
plt.grid(True, linestyle = '--', color = 'purple', alpha=0.5)
plt.hist(zena['f12'], color='red', alpha=0.4, label='Žena', density=False)
plt.hist(muz['f12'], color='purple', alpha=0.4, label='Muž', density=False)
plt.xlabel("f12 [cm]", fontsize=10)
plt.ylabel("Zastoupení (počet jedinců)", fontsize=10)
plt.legend(fontsize=10)
plt.show



