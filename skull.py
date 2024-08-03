import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import definice_funkci as dfu
import matplotlib.pyplot as plt
from scipy.stats import chi2
df = pd.read_excel("data_git.xlsx", sheet_name="skull final") 
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1})
sloupce = df.columns[2:]
presnosti_list = []
koeficienty_list = []
metriky_list = []
presnosti_train = []
presnosti_test = []
presnosti_sloupec_train = {sloupec_: [] for sloupec_ in sloupce}
for i in range(len(sloupce)):
    for j in range(i + 1, len(sloupce)):
        dvojice_sloupcu = [sloupce[i], sloupce[j]]
        # vezme každou dvojici sloupců a s ní pracuje
        data = df[['sex'] + dvojice_sloupcu]
        data.dropna(subset=dvojice_sloupcu + ['sex'], inplace=True)  
        y = data['sex'].values
        X = data[dvojice_sloupcu].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_pred = np.column_stack([np.ones(len(X_train)), X_train])
        ################################################## 
        #                                                #
        #            JEDNOTLIVÉ STATISTIKY               #
        #                                                #
        ##################################################
        koeficienty, shrnuti = dfu.model_logit(y_train, X_train)
        predikce_train = dfu.y_model(X_train_pred, koeficienty)
        binarizovane_predikce_train = (predikce_train > 0.5).astype(int) #rozdělení na 0  1
        shodne_hodnoty_train = (binarizovane_predikce_train == y_train)
        pocet_shodnych_zen_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 1)) #správně určené ženy
        pocet_shodnych_muzu_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 0)) #správně určení muži
        pocet_spatne_zen_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 1))
        pocet_spatne_muzu_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 0))
        presnost_train = round(dfu.presnost(pocet_shodnych_zen_train, pocet_shodnych_muzu_train, len(y_train)),2)
        odchylky = np.round(dfu.odchylka_log(X_train_pred, predikce_train),2)
        walduv_test = np.round(dfu.walduv_test(X_train_pred, predikce_train, koeficienty),2)
        p_hodnoty = np.round(dfu.p_hodnota_log(X_train_pred, predikce_train, koeficienty),2)
        presnosti_train.append(presnost_train)
        pocet_zen = data['sex'].value_counts()[1] #celkový počet žen
        pocet_muzu = data['sex'].value_counts()[0] #celkový počet mužů
        
        X_test_pred =  np.column_stack([np.ones(len(X_test)), X_test])
        predikce_test =  dfu.y_model(X_test_pred, koeficienty)
        binarizovane_predikce_test = (predikce_test > 0.5).astype(int)
        shodne_hodnoty_test = (binarizovane_predikce_test == y_test)
        pocet_shodnych_zen_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 1))
        pocet_spatne_zen_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 1))
        pocet_shodnych_muzu_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 0))
        pocet_spatne_muzu_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 0))
        presnost_test = round(dfu.presnost(pocet_shodnych_zen_test, pocet_shodnych_muzu_test, len(y_test)),2)
        presnosti_test.append(presnost_test)
        nagelkerke = round(dfu.nagelkerke(X_train_pred, y_train, koeficienty),2)
        mcFadden = round(dfu.mcFadden(X_train_pred, y_train, koeficienty),2)
        log_likehood = round(dfu.log_likelihood(X_train_pred, y_train, koeficienty),2)
        log_likehood_0 = round(dfu.log_likelihood_0((y_train)),2)
        f_1 = round(dfu.f1_skore(pocet_shodnych_zen_test+pocet_shodnych_zen_train, pocet_spatne_zen_test+pocet_spatne_zen_train, pocet_spatne_muzu_test+pocet_spatne_muzu_train),2)
        statistika = round(dfu.lrt(X_train_pred, y_train, koeficienty),2)
        odchylky = np.round(dfu.odchylka_log(X_train, predikce_train),2)
        interval_spol = np.round(dfu.interval_spolehlivosti_log(X_train_pred, predikce_train, koeficienty),2)
        if all(p <= 0.05 for p in p_hodnoty):
            vyznamnost = "model je významný"
        else: vyznamnost = "nelze zamítnout hypotézy"
        if statistika > chi2.ppf(1 - 0.05, 2):
            lrt_test = "Zamítáme nulovou hypotézu, úplný model je lepší"
        else:lrt_test = "Nezamítáme nulovou hypotézu, plný model není lepší než redukovaný model."
        if dvojice_sloupcu not in [sloupec[3] for sloupec in presnosti_list]:
                    presnosti_list.append((presnost_train, presnost_test, f_1, dvojice_sloupcu, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu))
        if dvojice_sloupcu not in [sloupec[2] for sloupec in koeficienty_list]:
                    koeficienty_list.append((presnost_train, presnost_test, dvojice_sloupcu, koeficienty[0], koeficienty[1], koeficienty[2], p_hodnoty[0], p_hodnoty[1], p_hodnoty[2], walduv_test[0], walduv_test[1], walduv_test[2], vyznamnost))
        if dvojice_sloupcu not in [sloupec[2] for sloupec in metriky_list]:
                     metriky_list.append((presnost_train, presnost_test, dvojice_sloupcu,log_likehood, log_likehood_0, nagelkerke, mcFadden, statistika, statistika))
        for sloupec in dvojice_sloupcu:
            presnosti_sloupec_train[sloupec].append(presnost_train)
        ################################################## 
        #                                                #
        #                 VÝPIS VÝSLEDKŮ                 #
        #                                                #
        ##################################################       
        #print(shrnuti)
        print(f"\nLogistická regrese pro sloupce {dvojice_sloupcu}:")
        print("Koeficienty:\n", koeficienty)
        print("celkový počet mužů je:", pocet_muzu)
        print("celkový počet žen je:", pocet_zen)
        print("\nPočet správně určených žen (trénovací množina):", pocet_shodnych_zen_train)
        print("\nPočet správně určených mužů (trénovací množina):", pocet_shodnych_muzu_train)
        print("\nPřesnost pro sloupce (trénovací množina) je:", presnost_train)
        print("Směrodatné odchylky koeficientů:", dfu.odchylka_log(X_train_pred, predikce_train))
        print("p_hodnoty pro koeficienty:", p_hodnoty)
        print("Intervaly spolehlivosti pro jednotlivé koeficienty:", dfu.interval_spolehlivosti_log(X_train_pred, predikce_train, koeficienty))
        print("Počet správně určených žen (testovací množina):", pocet_shodnych_zen_test)
        print("\nPočet správně určených mužů (testovací množina):", pocet_shodnych_muzu_test)
        print("\nPřesnost pro sloupce (testovací množina) je:", presnost_test)
        print("Waldova statistika pro jednotlivé koeficienty",walduv_test)
        print("Odchylky jednotlivých koeficientů", odchylky)
        print("Jednotlivé intervaly spolehlivosti koeficientů", interval_spol)
        print("Testy statistické významnosti jednotlivých koeficientů ukázaly, že",vyznamnost)
        print("Dle LRT testu",lrt_test)
################################################## 
#                                                #
#                   GRAFY                        #
#                                                #
##################################################
#krabicový graf
presnosti_train_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in presnosti_sloupec_train.items()]))
presnosti_train_df = presnosti_train_df.melt(var_name='Měření', value_name='Přesnost')
median = presnosti_train_df.groupby('Měření')['Přesnost'].median()
serazene_dle_medianu = median.sort_values().index
presnosti_train_df['Měření'] = pd.Categorical(presnosti_train_df['Měření'], categories=serazene_dle_medianu, ordered=True)
plt.figure(figsize=(22, 10))
ax = plt.gca()
presnosti_train_df.boxplot(column='Přesnost', by='Měření', ax=ax)
plt.xticks(rotation=90)
plt.xlabel('Měření', fontsize=10)
plt.ylabel('Přesnost trénovací množiny', fontsize=10)
plt.yticks(np.arange(0.4, 1.01, 0.05))
plt.suptitle('')
plt.gca().set_title('')
plt.show()
#graf křivek přesností
x_hodnoty = np.arange(0, 1.01, 0.01)
y_hodnoty_train = [np.sum(np.array(presnosti_train) >= x) for x in x_hodnoty]
y_hodnoty_test = [np.sum(np.array(presnosti_test) >= z) for z in x_hodnoty]
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x_hodnoty, y_hodnoty_train, linestyle='-', color='blue', label="Trénovací přesnost")
ax1.plot(x_hodnoty, y_hodnoty_test, linestyle='-', color='red', label="Testovací přesnost")
ax1.set_xlabel('Přesnost', fontsize=10)
ax1.set_ylabel('Počet modelů s přesností vyšší než na ose x', fontsize=10)
ax1.set_xticks(np.arange(0, 1.005, 0.1))
ax1.grid(True)
ax1.set_xlim(0, 1)
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
#                  TABULKY                       #
#                                                #
################################################## 
#tabulka přesností
presnosti_list.sort(reverse=True)
sloupce = ['Přesnost (train)', 'Přesnost (test)','F-1 skóre', 'Kódy kostí', 'Počet správně odhadnutých žen (train)', 'Počet správně odhadnutých mužů (train)', 'Počet správně odhadnutých žen (test)', 'Počet správně odhadnutých mužů (test)','Počet žen', 'Počet mužů']
data_tab = [(round(presnost_train,2), round(presnost_test,2), round(f_1,2), sloupce, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu) for presnost_train, presnost_test,f_1, sloupce, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test,pocet_zen, pocet_muzu in presnosti_list]
df_sloupce = pd.DataFrame(data_tab, columns=sloupce)
df_filtr_accuracy = df_sloupce.sort_values(by=['Přesnost (train)', 'Přesnost (test)','Počet správně odhadnutých žen (train)', 'Počet správně odhadnutých mužů (train)'], ascending=False) #seřazení podle nejvyššího trénovací a testovací přesnosti
plt.figure(figsize=(12, 6))
table = plt.table(cellText=df_filtr_accuracy.head(50).values, colLabels=df_filtr_accuracy.columns, loc='center', cellLoc='center', colWidths=[0.1,0.3, 0.3, 0.2, 0.4, 0.4, 0.4, 0.4,0.2,0.2]) 
table.set_fontsize(12)
table.scale(1.2, 3.5)  
plt.axis('off')
plt.show()
#tabulka koeficientů
koeficienty_list.sort(reverse = True)
sloupce_koef = ['Přesnost (train)', 'Přesnost (test)', 'Kódy kostí', '\u03B20', '\u03B21', '\u03B22', 'p hodnota \u03B20', 'p hodnota \u03B21', 'p hodnota \u03B22', 'Wald \u03B20', 'Wald \u03B21 ', 'Wald \u03B22', 'Statistická významnost']
data_tab2 = [(round(presnost_train,2), round(presnost_test,2), sloupce, round(koeficienty[0],2), round(koeficienty[1],2), round(koeficienty[2],2), round(p_hodnoty[0],2), round(p_hodnoty[1],2), round(p_hodnoty[2],2), round(walduv_test[0],2), round(walduv_test[1],2), round(walduv_test[2],2), vyznamnost) for presnost_train, presnost_test, sloupce, koeficienty[0], koeficienty[1], koeficienty[2], p_hodnoty[0], p_hodnoty[1], p_hodnoty[2], walduv_test[0], walduv_test[1], walduv_test[2],vyznamnost in koeficienty_list]
df_sloupce2 = pd.DataFrame(data_tab2, columns=sloupce_koef)
df_filtr_train2 = df_sloupce2.sort_values(by=['Přesnost (train)', 'Přesnost (test)'], ascending=False) #seřazení podle nejvyššího počtu správně určených žen a mužů
plt.figure(figsize=(22, 16))
table = plt.table(cellText=df_filtr_train2.head(30).values, colLabels=df_filtr_train2.columns, loc='center', cellLoc='center', colWidths=[0.1, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2,0.2, 0.2, 0.2,0.2, 0.2, 0.2, 0.2])
table.set_fontsize(12)
table.scale(1.2, 3.5)  
plt.axis('off')
plt.show()
#tabulka ostatních metrik
metriky_list.sort(reverse = True)
sloupce_metriky = [ 'Přesnost (train)', 'Přesnost (test)', 'Sloupce', 'Loglikelihood ', 'Loglikelihood 0', 'Nagelkerkeho koeficient', 'Koeficient R2', 'LRT statistika', 'Výsledek LRT testu']
data_tab3 = [(presnost_train,presnost_test,dvojice_sloupcu, log_likehood, log_likehood_0, nagelkerke, mcFadden, statistika, lrt_test) for presnost_train, presnost_test, dvojice_sloupce, log_likehood, log_likehood_0, nagelkerke, mcFadden, statistika, lrt_test in metriky_list]
df_sloupce3 = pd.DataFrame(data_tab3, columns=sloupce_metriky)
plt.figure(figsize=(22, 16))
table = plt.table(cellText=df_sloupce3.head(150).values, colLabels=df_sloupce3.columns, loc='center', cellLoc='center', colWidths=[0.1, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.3])
table.set_fontsize(12)
table.scale(1.2, 3.5)  
plt.axis('off')
plt.show()
#spojení tabulek do jedné excelové
tabulka = pd.concat([df_sloupce, df_sloupce2, df_sloupce3], axis=1)
#tabulka.to_excel('skull__vysledky.xlsx', index=False)

