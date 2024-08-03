import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import definice_funkci as dfu
from scipy.stats import f
from scipy.stats import chi2
df = pd.read_excel("data_git.xlsx", sheet_name="postcranial final_corr")
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1})

sloupce = df.columns[2:20]
presnosti_list = []
for i in range(len(sloupce)):
        sloupec = [sloupce[i]]
        data = df[['sex'] + sloupec]
        data = data.dropna(subset=sloupec, how='any')
        y = data['sex'].values
        X = data[sloupec].values
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
          outlier = dfu.outlier_log(X_train_pred, y_train, koeficienty)
          leverage = dfu.leverage_point_log(X_train_pred, koeficienty)
          cook_vzdalenosti = dfu.cook_vzd_log(X_train_pred, y_train, koeficienty)
          influent_nalezen = False
          outlier_nalezen = False
          leverage_nalezen = False
          predikce_train =  dfu.y_model(X_train_pred, koeficienty)
          binarizovane_predikce_train = (predikce_train > 0.5).astype(int)
          shodne_hodnoty_train = (binarizovane_predikce_train == y_train)
          pocet_shodnych_zen_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 1))
          pocet_shodnych_muzu_train = np.sum((shodne_hodnoty_train == True) & (binarizovane_predikce_train == 0))
          presnost_train = round(dfu.presnost(pocet_shodnych_zen_train, pocet_shodnych_muzu_train, len(predikce_train)),2)
          pocet_spatne_zen_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 1))
          pocet_spatne_muzu_train = np.sum((shodne_hodnoty_train == False) & (binarizovane_predikce_train == 0))

          X_test_pred = np.column_stack([np.ones(len(X_test)), X_test])
          predikce_test =  dfu.y_model(X_test_pred, koeficienty)
          binarizovane_predikce_test = (predikce_test > 0.5).astype(int)
          shodne_hodnoty_test = (binarizovane_predikce_test == y_test)
          pocet_shodnych_zen_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 1))
          pocet_shodnych_muzu_test = np.sum((shodne_hodnoty_test == True) & (binarizovane_predikce_test == 0))
          pocet_muzu =  data['sex'].value_counts().get(0, 0)
          pocet_zen = data['sex'].value_counts().get(1, 0)
          presnost_test = round(dfu.presnost(pocet_shodnych_zen_test, pocet_shodnych_muzu_test, len(y_test)),2)
          pocet_spatne_zen_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 1))
          pocet_spatne_muzu_test = np.sum((shodne_hodnoty_test == False) & (binarizovane_predikce_test == 0))
          f_1_test = round(dfu.f1_skore(pocet_shodnych_zen_test, pocet_spatne_zen_test, pocet_spatne_muzu_test),2)
          p_hodnoty = np.round(dfu.p_hodnota_log(X_train_pred, predikce_train, koeficienty),2)
          walduv_test = np.round(dfu.walduv_test(X_train_pred, predikce_train, koeficienty),2)
          f_1 = round(dfu.f1_skore(pocet_shodnych_zen_test+pocet_shodnych_zen_train, pocet_spatne_zen_test+pocet_spatne_zen_train, pocet_spatne_muzu_test+pocet_spatne_muzu_train),2)
          log_likehood_0 = round(dfu.log_likelihood_0(y_train),2)
          log_likehood = round(dfu.log_likelihood(X_train_pred, y_train, koeficienty),2)
          mcFadden = round(dfu.mcFadden(X_train_pred, y_train, koeficienty),2)
          nagelkerke = round(dfu.nagelkerke(X_train_pred, y_train, koeficienty),2)
          if all(p <= 0.05 for p in p_hodnoty):
                   vyznamnost = "Model je statisticky významný"
          else: vyznamnost ="Nelze zamítnout nulové hypotézy"
          statistika = round(dfu.lrt(X_train_pred, y_train, koeficienty),2)
          if statistika > chi2.ppf(1 - 0.05, 2):
            lrt_test = "Zamítáme nulovou hypotézu, úplný model je lepší"
          else: lrt_test = "Nezamítáme nulovou hypotézu, plný model není lepší než redukovaný model."
          if sloupec not in [sloupec_i[2] for sloupec_i in presnosti_list]:
           presnosti_list.append((presnost_train, presnost_test, sloupec, f_1, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu, koeficienty[0], koeficienty[1],  p_hodnoty[0], p_hodnoty[1],  walduv_test[0], walduv_test[1],  vyznamnost, log_likehood_0, log_likehood, nagelkerke, mcFadden, statistika, lrt_test))
          ################################################## 
          #                                                #
          #                 VÝPIS VÝSLEDKŮ                 #
          #                                                #
          ##################################################
          print(dfu.vif(X_train))
          #print(shrnuti)
          print(f"\nLogistická regrese pro sloupec {sloupec}:")
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
          if not influent_nalezen: print("influent nenalezen") 

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
          print(f"Singulární matice pro sloupec {sloupec}: {e}")
          continue 
################################################## 
#                                                #
#                   TABULKY                      #
#                                                #
##################################################
presnosti_list.sort(reverse=True)
sloupce = ['Přesnost (train)', 'Přesnost (test)', 'Sloupce', 'F-1 skóre','Počet správně odhadnutých žen (train)', 'Počet správně odhadnutých mužů (train)', 'Počet správně odhadnutých žen (test)', 'Počet správně odhadnutých mužů (test)','Počet žen', 'Počet mužů', '\u03B20', '\u03B21', 'p hodnota \u03B20', 'p hodnota \u03B21',  'Wald \u03B20', 'Wald \u03B21', 'Významnost', 'Loglikelihood 0', 'Loglikelihood', 'Nagelkerkeho koeficient', 'Koeficient R2', 'Statistika LRT', 'Výsledek LRT testu']
data_tab = [(presnost_train, presnost_test,sloupec, f_1, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu, koeficienty[0], koeficienty[1],  p_hodnoty[0], p_hodnoty[1],  walduv_test[0], walduv_test[1],  vyznamnost, log_likehood_0, log_likehood, nagelkerke, R_log, statistika, lrt_test) for presnost_train, presnost_test,sloupec,f_1, pocet_shodnych_zen_train, pocet_shodnych_muzu_train, pocet_shodnych_zen_test, pocet_shodnych_muzu_test, pocet_zen, pocet_muzu, koeficienty[0], koeficienty[1],  p_hodnoty[0], p_hodnoty[1],  walduv_test[0], walduv_test[1],  vyznamnost, log_likehood_0, log_likehood, nagelkerke, R_log, statistika, lrt_test in presnosti_list]
df_sloupce = pd.DataFrame(data_tab, columns=sloupce)
df_filtr_train = df_sloupce.sort_values(by=['Počet správně odhadnutých žen (train)', 'Počet správně odhadnutých mužů (train)'], ascending=False) #seřazení podle nejvyššího počtu správně určených žen a mužů
tabulka = pd.concat([df_sloupce], axis=1)
#tabulka.to_excel('postcr_vysledky_1rozmer.xlsx', index=False)
