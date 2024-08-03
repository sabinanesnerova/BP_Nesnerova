import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import definice_funkci as dfu
import matplotlib.pyplot as plt
from scipy.stats import f, chi2
df = pd.read_excel("data_git.xlsx", sheet_name="postcranial final_corr") 
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1})
data = df[['sex','f6', 'f12']] #lze volit libovolnou dvojici v libovolném listu
data = data.dropna(subset=['sex', 'f6', 'f12'], how='any')  # odstranění řádků s chybějícími hodnotami
y = data['sex'].values
X = data[['f6', 'f12']].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
koeficienty, shrnuti = dfu.model_logit(y_train, X_train)
koeficienty = np.round(koeficienty, 2)
X_train_pred = np.column_stack([np.ones(len(X_train)), X_train])
predikce_train =  dfu.y_model(X_train_pred, koeficienty) 
#identifikace odlehlých bodů
outlier = dfu.outlier_log(X_train_pred, y_train, koeficienty)
leverage = dfu.leverage_point_log(X_train_pred, koeficienty)
cook_vzdalenosti = dfu.cook_vzd_log(X_train_pred, y_train, koeficienty)
influent_nalezen = False
outlier_nalezen = False
leverage_nalezen = False
for k, cook_vzd in enumerate(cook_vzdalenosti):
       if cook_vzd > f.ppf(0.5, len(koeficienty), len(y_train)-len(koeficienty)):
            print(f"bod {X_train_pred[k]}, {y_train[k]} je influentní")
            influent_nalezen = True
if not influent_nalezen:
    print("influent nenalezen") 
for k, lev in enumerate(leverage):
       if lev ==True:
            print(f"bod {X_train_pred[k]}, {y_train[k]} je leverage")
            leverage_nalezen = True
if not leverage_nalezen: print("leverage nenalezen") 
for k, out in enumerate(outlier):
       if out ==True:
            print(f"bod {X_train_pred[k]}, {y_train[k]} je outlier")
            outlier_nalezen = True
if not outlier_nalezen: print("outlier nenalezen") 
################################################## 
#                                                #
#             JEDNOTLIVÉ STATISTIKY              #
#                                                #
##################################################
binarizovane_predikce_train = (predikce_train > 0.5).astype(int) #rozdělení na 0  1
shodne_hodnoty_train = (binarizovane_predikce_train == y_train)
pocet_shodnych_zen_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 1)) #správně určené ženy
pocet_shodnych_muzu_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 0)) #správně určení muži
pocet_spatne_zen_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 1))
pocet_spatne_muzu_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 0))
presnost_train = round(dfu.presnost(pocet_shodnych_zen_train, pocet_shodnych_muzu_train, len(y_train)),2)
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
f1 = round(dfu.f1_skore(pocet_shodnych_zen_test+pocet_shodnych_zen_train, pocet_spatne_zen_test+pocet_spatne_zen_train, pocet_spatne_muzu_test+pocet_spatne_muzu_train),2)
nagelkerke = round(dfu.nagelkerke(X_train_pred, y_train, koeficienty),2)
vif = dfu.vif(X_train_pred)
mcFadden = round(dfu.mcFadden(X_train_pred, y_train, koeficienty),2)
odchylky = np.round(dfu.odchylka_log(X_train_pred, predikce_train),2)
p_hodnoty = np.round(dfu.p_hodnota_log(X_train_pred, predikce_train, koeficienty),2)
intervaly_spol = np.round(dfu.interval_spolehlivosti_log(X_train_pred, predikce_train, koeficienty),2)
deviance = round(dfu.deviance(X_train_pred, y_train, koeficienty),2)
log_likelihood = round(dfu.log_likelihood(X_train_pred, y_train, koeficienty),2)
log_likelihood0 = round(dfu.log_likelihood_0(y_train),2)
if all(p <= 0.05 for p in p_hodnoty):
         vyznamnost = "Model je statisticky významný"
else: vyznamnost ="Nelze zamítnout nulové hypotézy"
statistika = round(dfu.lrt(X_train_pred, y_train, koeficienty),2)
if statistika > chi2.ppf(1 - 0.05, len(y_train)-len(koeficienty)):
  lrt_test = "Zamítáme nulovou hypotézu, úplný model je lepší"
else: lrt_test = "Nezamítáme nulovou hypotézu, plný model není lepší než redukovaný model."
###############################################
#                                             #
#             VÝPIS VÝSLEDKŮ                  #
#                                             #
###############################################
print("Koeficienty:\n", koeficienty)
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
print("Nagelkerkeho koeficient kvality je:", nagelkerke)
print("Mc Faddenův koeficient kvality je:", mcFadden)
print("Koeficient detekující multikolinearitu je:", vif)
print("Deviance modelu je:", deviance)
print("F-1 skóre je", f1)
print("Waldův test pro jednotlivé koeficienty je:",walduv_test)
print("Věrohodnostní logaritmická funkce modelu je", log_likelihood)
print("Věrohodnostní logaritmická funkce modelu pouze s konstantním koeficientem je", log_likelihood)
print("Testy statistické významnosti jednotlivých koeficientů ukázaly, že",vyznamnost)
print("Dle LRT testu",lrt_test)
###############################################
#                                             #
#                  GRAFY                      #
#                                             #
###############################################
#vykreslení rozložení dat
zena_data = data[data['sex'] == 1]
muz_data = data[data['sex'] == 0]
plt.figure(figsize=(10, 6))
plt.scatter(zena_data['f6'], zena_data['f12'], c='red', marker='o', label='Žena')
plt.scatter(muz_data['f6'], muz_data['f12'], c='blue', marker='x', label='Muž')
plt.xlabel('f6', fontsize=10)
plt.ylabel('f12', fontsize=10)
plt.legend(fontsize=10)
plt.show()
#vykreslení nadroviny
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
hodnoty_1_mezi_hranicemi = np.linspace(X[:, 0].min(), X[:, 0].max(), num=100)
realne_hodnoty_1 = (X[:, 0])
x_hodnoty_1 = (np.unique(np.concatenate((hodnoty_1_mezi_hranicemi, realne_hodnoty_1))))
hodnoty_2_mezi_hranicemi = np.linspace(X[:, 1].min(), X[:, 1].max(), num=100)
realne_hodnoty_2 = (X[:, 1])
x_hodnoty_2 = (np.unique(np.concatenate((hodnoty_2_mezi_hranicemi, realne_hodnoty_2))))
x_mrizka_1, x_mrizka_2 = np.meshgrid(x_hodnoty_1, x_hodnoty_2)
X_mrizka = np.column_stack([np.ones(x_mrizka_1.size),x_mrizka_1.ravel(), x_mrizka_2.ravel()])
predikovane_hodnoty = dfu.y_model(X_mrizka, koeficienty).reshape(x_mrizka_1.shape)
zena_X_train = X_train[y_train == 1]
muz_X_train = X_train[y_train == 0]
ax.scatter(zena_X_train[:,0], zena_X_train[:,1],1, c='red', marker='o', label='Žena', alpha = 1)
ax.scatter(muz_X_train[:,0], muz_X_train[:,1],0, c='blue', marker='o', label='Muž', alpha=1)
surface = ax.plot_surface(x_mrizka_1, x_mrizka_2, predikovane_hodnoty, cmap='coolwarm', alpha=0.6)
ax.set_xlabel('f6', fontsize=8,labelpad=6)
ax.set_ylabel('f12', fontsize=8, labelpad=6)
boundaries = np.linspace(0, 1, 6)
cbar = plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5, boundaries=boundaries)
cbar.set_label('Pravděpodobnost muž - žena', fontsize=8)
cbar.ax.tick_params(labelsize=8) 
ax.view_init(elev=25, azim=-23)
plt.tick_params(axis='both', which='major', labelsize=8)  
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.show()


