import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import definice_funkci as dfu
from scipy.stats import f
from scipy.stats import chi2
df = pd.read_excel("data_git.xlsx", sheet_name="postcranial final_corr")
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1})
sloupce = df.columns[2:20]
print(sloupce)
presnosti_list = []
koeficienty_list = []
metriky_list = []
presnosti_train = []
presnosti_test = []
presnosti_sloupec_train = {sl: [] for sl in sloupce}
for i in range(len(sloupce)):
    for j in range(i+1, len(sloupce)):
        dvojice_sloupcu = [sloupce[i], sloupce[j]]
        data = df[['sex'] + dvojice_sloupcu]
        data = data.dropna(subset=dvojice_sloupcu, how='any')
        y = data['sex'].values
        X = data[dvojice_sloupcu].values
        if len(y) < 10:
           print(f"Přeskočena dvojice {dvojice_sloupcu}, protože není dostatečně velký vzorek")
           continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Rozdělení dat na trénovací a testovací množinu (80% trénovací, 20% testovací)
        X_train_pred = np.column_stack([np.ones(len(X_train)), X_train])  
        try:
          ################################################## 
          #                                                #
          #             JEDNOTLIVÉ STATISTIKY              #
          #                                                #
          ##################################################
          koeficienty, shrnuti = dfu.model_logit(y_train, X_train)
          koeficienty = np.round(koeficienty,2)
          predikce_train =  dfu.y_model(X_train_pred, koeficienty)
          binarizovane_predikce_train = (predikce_train > 0.5).astype(int)
          shodne_hodnoty_train = (binarizovane_predikce_train == y_train)
          pocet_shodnych_zen_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 1))
          pocet_shodnych_muzu_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 0))
          presnost_train = round(dfu.presnost(pocet_shodnych_zen_train, pocet_shodnych_muzu_train, len(predikce_train)),2)
          pocet_spatne_zen_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 1))
          pocet_spatne_muzu_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 0))
          presnosti_train.append(presnost_train)
          
          X_test_pred = np.column_stack([np.ones(len(X_test)), X_test])
          predikce_test =  dfu.y_model(X_test_pred, koeficienty)
          binarizovane_predikce_test = (predikce_test > 0.5).astype(int)
          shodne_hodnoty_test = (binarizovane_predikce_test == y_test)
          pocet_shodnych_zen_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 1))
          pocet_shodnych_muzu_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 0))
          pocet_muzu =  data['sex'].value_counts().get(0, 0)
          pocet_zen = data['sex'].value_counts().get(1, 0)
          presnost_test = round(dfu.presnost(pocet_shodnych_zen_test, pocet_shodnych_muzu_test, len(y_test)),2)
          presnosti_test.append(presnost_test)
          pocet_spatne_zen_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 1))
          pocet_spatne_muzu_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 0))
          p_hodnoty = np.round(dfu.p_hodnota_log(X_train_pred, predikce_train, koeficienty),2)
          walduv_test = np.round(dfu.walduv_test(X_train_pred, predikce_train, koeficienty),2)
          f_1 = round(dfu.f1_skore(pocet_shodnych_zen_test+pocet_shodnych_zen_train, pocet_spatne_zen_test+pocet_spatne_zen_train, pocet_spatne_muzu_test+pocet_spatne_muzu_train),2)
          log_likehood_0 = round(dfu.log_likelihood_0(y_train),2)
          log_likehood = round(dfu.log_likelihood(X_train_pred, y_train, koeficienty),2)
          mcFadden = round(dfu.mcFadden(X_train_pred, y_train, koeficienty),2)
          nagelkerke = round(dfu.nagelkerke(X_train_pred, y_train, koeficienty),2)
          outlier = dfu.outlier_log(X_train_pred, y_train, koeficienty)
          leverage = dfu.leverage_point_log(X_train_pred, koeficienty)
          cook_vzdalenosti = dfu.cook_vzd_log(X_train_pred, y_train, koeficienty)
          influent_nalezen = False
          outlier_nalezen = False
          leverage_nalezen = False
          if all(p <= 0.05 for p in p_hodnoty):
                   vyznamnost = "Model je statisticky významný"
          else: vyznamnost ="Nelze zamítnout nulové hypotézy"
          statistika = round(dfu.lrt(X_train_pred, y_train, koeficienty),2)
          if statistika > chi2.ppf(1 - 0.05, len(y_train)-len(koeficienty)):
            lrt_test = "Zamítáme nulovou hypotézu, úplný model je lepší"
          else: lrt_test = "Nezamítáme nulovou hypotézu, plný model není lepší než redukovaný model."
          if dvojice_sloupcu not in [sloupec[2] for sloupec in presnosti_list]:
           presnosti_list.append((presnost_train, presnost_test, dvojice_sloupcu, f_1, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu))
          if dvojice_sloupcu not in [sloupec[2] for sloupec in koeficienty_list]:
           koeficienty_list.append((presnost_train, presnost_test, dvojice_sloupcu, koeficienty[0], koeficienty[1], koeficienty[2], p_hodnoty[0], p_hodnoty[1], p_hodnoty[2], walduv_test[0], walduv_test[1], walduv_test[2], vyznamnost))
          if dvojice_sloupcu not in [sloupec[2] for sloupec in metriky_list]:
           metriky_list.append((presnost_train, presnost_test, dvojice_sloupcu,log_likehood_0,log_likehood, nagelkerke, mcFadden, statistika, lrt_test))
          for sloupec in dvojice_sloupcu:
                  presnosti_sloupec_train[sloupec].append(presnost_train)
          ################################################## 
          #                                                #
          #                VÝPIS VÝSLEDKŮ                  #
          #                                                #
          ##################################################
          print(dfu.vif(X_train))
          print(f"\nLogistická regrese pro sloupce {dvojice_sloupcu}:")
          print("Koeficienty:", koeficienty)
          print("Počet správně určených žen (trénovací množina):", pocet_shodnych_zen_train)
          print("Počet správně určených mužů (trénovací množina):", pocet_shodnych_muzu_train)
          print("Přesnost pro sloupce (trénovací množina) je:", presnost_train)
          print("Směrodatné odchylky koeficientů:", dfu.odchylka_log(X_train_pred, predikce_train))
          print("p_hodnoty pro koeficienty:", dfu.p_hodnota_log(X_train_pred, predikce_train, koeficienty))
          print("Intervaly spolehlivosti pro jednotlivé koeficienty:", dfu.interval_spolehlivosti_log(X_train_pred, predikce_train, koeficienty))
          print("Waldův test pro jednotlivé koeficienty:",walduv_test)
          print("Počet správně určených žen (testovací množina):", pocet_shodnych_zen_test)
          print("Počet správně určených mužů (testovací množina):", pocet_shodnych_muzu_test)
          print("Přesnost pro sloupce (testovací množina) je:", presnost_test)
          print("Celkový počet mužů je:",pocet_muzu)
          print("Celkový počet žen je:", pocet_zen)
          print("Testy statistické významnosti jednotlivých koeficientů ukázaly, že",vyznamnost)
          print("Dle LRT testu",lrt_test)
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
        except np.linalg.LinAlgError as e:
          print(f"Singulární matice pro sloupce {dvojice_sloupcu}: {e}")
          continue  
################################################## 
#                                                #
#                    GRAFY                       #
#                                                #
##################################################
#krabicový graf
presnosti_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in presnosti_sloupec_train.items()]))
presnosti_df = presnosti_df.melt(var_name='Měření', value_name='Přesnost')
median = presnosti_df.groupby('Měření')['Přesnost'].median()
serazene_dle_medianu = median.sort_values().index
presnosti_df['Měření'] = pd.Categorical(presnosti_df['Měření'], categories=serazene_dle_medianu, ordered=True)
plt.figure(figsize=(25, 10))
ax = plt.gca()
presnosti_df.boxplot(column='Přesnost', by='Měření', ax=ax)
plt.xticks(rotation=90)
plt.xlabel('Měření', fontsize = 10)
plt.ylabel('Přesnost trénovací množiny', fontsize = 10)
plt.yticks(np.arange(0.6, 1.01, 0.05)) 
plt.suptitle('')
plt.gca().set_title('') 
plt.show()
#graf křivek přesností
x_hodnoty = np.arange(0, 1.01, 0.01)
y_hodnoty_train= [np.sum(np.array(presnosti_train) >= x) for x in x_hodnoty]
y_hodnoty_test = [np.sum(np.array(presnosti_test) >= z) for z in x_hodnoty]
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x_hodnoty, y_hodnoty_train, linestyle='-', color='blue', label="Trénovací přesnost")
ax1.plot(x_hodnoty, y_hodnoty_test, linestyle='-', color='red', label="Testovací přesnost")
ax1.set_xlabel('Přesnost', fontsize=10)
ax1.set_ylabel('Počet modelů s přesností vyšší než na ose x', fontsize=10)
ax1.set_xticks(np.arange(0, 1.005, 0.05))
ax1.grid(True)
ax1.set_xlim(0.6, 1,0.05)
ax1.set_ylim(0, max(y_hodnoty_train) * 1.05)
ax2 = ax1.twinx()
ax2.set_ylabel('Procenta modelů s přesností vyšší než na ose x', fontsize=10)
max_y = max(y_hodnoty_train) 
ax2.set_ylim(0, 100)
yticks = np.linspace(0, 100, num=11)
ax2.set_yticks(yticks)
ax2.set_yticklabels([f'{int(ytick)}%' for ytick in yticks])
ax2.set_ylim(dfu.procenta_stupnice(ax1.get_ylim()[0], max_y), dfu.procenta_stupnice(ax1.get_ylim()[1], max_y))
fig.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.9, 0.85))
plt.show()    
################################################## 
#                                                #
#                    TABULKY                     #
#                                                #
##################################################
#tabulka přesností
presnosti_list.sort(reverse=True)
sloupce = ['Přesnost (train)', 'Přesnost (test)', 'Sloupce', 'F-1 skóre','Počet správně odhadnutých žen (train)', 'Počet správně odhadnutých mužů (train)', 'Počet správně odhadnutých žen (test)', 'Počet správně odhadnutých mužů (test)','Počet žen', 'Počet mužů']
data_tab = [(presnost_train, presnost_test,dvojice_sloupcu, f_1, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu) for presnost_train, presnost_test, dvojice_sloupcu,f_1, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu in presnosti_list]
df_sloupce = pd.DataFrame(data_tab, columns=sloupce)
df_filtr_train = df_sloupce.sort_values(by=['Počet správně odhadnutých žen (train)', 'Počet správně odhadnutých mužů (train)'], ascending=False) #seřazení podle nejvyššího počtu správně určených žen a mužů
plt.figure(figsize=(12, 6))
table = plt.table(cellText=df_filtr_train.head(150).values, colLabels=df_filtr_train.columns, loc='center', cellLoc='center', colWidths=[0.2, 0.2,0.2, 0.2, 0.4, 0.4, 0.4, 0.4,0.2, 0.2])
table.set_fontsize(12)
table.scale(1.2, 3.5)  
plt.axis('off')
plt.show()
#tabulka koeficientů
koeficienty_list.sort(reverse = True)
sloupce_koef = [ 'Přesnost (train)', 'Přesnost (test)', 'Sloupce', '\u03B20', '\u03B21', '\u03B22', 'p hodnota \u03B20', 'p hodnota \u03B21', 'p hodnota \u03B22', 'Wald \u03B20', 'Wald \u03B21', 'Wald \u03B22', 'Významnost']
data_tab2 = [(presnost_train,presnost_test, dvojice_sloupcu, koeficienty[0], koeficienty[1], koeficienty[2], p_hodnoty[0], p_hodnoty[1], p_hodnoty[2], walduv_test[0], walduv_test[1], walduv_test[2], vyznamnost) for presnost_train, presnost_test, dvojice_sloupcu, koeficienty[0], koeficienty[1], koeficienty[2], p_hodnoty[0], p_hodnoty[1], p_hodnoty[2], walduv_test[0], walduv_test[1], walduv_test[2], vyznamnost in koeficienty_list]
df_sloupce2 = pd.DataFrame(data_tab2, columns=sloupce_koef)
plt.figure(figsize=(22, 16))
table = plt.table(cellText=df_sloupce2.head(150).values, colLabels=df_sloupce2.columns, loc='center', cellLoc='center', colWidths=[0.1, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2,0.2, 0.2, 0.2,0.2, 0.2, 0.2, 0.2])
table.set_fontsize(12)
table.scale(1.2, 3.5)  
plt.axis('off')
plt.show()
#tabulka zbylých metrik
metriky_list.sort(reverse = True)
sloupce_metriky = [ 'Přesnost (train)', 'Přesnost (test)', 'Sloupce', 'Loglikelihood 0', 'Loglikelihood', 'Nagelkerkeho koeficient', 'McFaddenův koeficient', 'Statistika LRT', 'Výsledek LRT testu']
data_tab3 = [(presnost_train, presnost_test, dvojice_sloupcu, log_likehood_0, log_likehood, nagelkerke, mcFadden, statistika, lrt_test) for presnost_train, presnost_test, dvojice_sloupce, log_likehood_0, log_likehood, nagelkerke, mcFadden, statistika, lrt_test in metriky_list]
df_sloupce3 = pd.DataFrame(data_tab3, columns=sloupce_metriky)
plt.figure(figsize=(22, 16))
table = plt.table(cellText=df_sloupce3.head(150).values, colLabels=df_sloupce3.columns, loc='center', cellLoc='center', colWidths=[0.1, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
table.set_fontsize(12)
table.scale(1.2, 3.5)  
plt.axis('off')
plt.show()
#sjednocení tabulek do jedné excelové
tabulka = pd.concat([df_sloupce, df_sloupce2, df_sloupce3], axis=1)
#tabulka.to_excel('postcran_vysledky.xlsx', index=False)

