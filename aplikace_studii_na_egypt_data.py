import pandas as pd
import numpy as np
import definice_funkci as dfu
df = pd.read_excel("data_git.xlsx", sheet_name="postcranial final_corr") 
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1}) #binarizace pohlaví
sloupce = df.columns[4:] #bere sloupce od 4.dál
####################################
#                                  #
#             Dabbs                #
#                                  #
####################################
data = df[['sex','f18', 'f19']]
data = data.dropna(subset=['sex', 'f18', 'f19'], how='any')  # Odstranění řádků s prázdnými buňkami
y = data['sex'].values
X = data[['f18', 'f19']].values  
koeficienty = [-35.834, 0.674, 0.493] #koeficienty získané ze studie
X_pred = np.column_stack([np.ones(len(X)), X])
predikce = dfu.rovnice(X_pred, koeficienty)  #pravděpodobnost, že se jedná o ženu
binarizovane_predikce= (predikce < 0).astype(int) #rozdělení na 0  1 pomocí diskriminačního skóre
shodne_hodnoty = (binarizovane_predikce == y)
pocet_shodnych_zen = np.sum((shodne_hodnoty == True) & (binarizovane_predikce == 1)) #správně určené ženy
pocet_shodnych_muzu = np.sum((shodne_hodnoty == True) & (binarizovane_predikce == 0)) #správně určení muži
presnost = dfu.presnost(pocet_shodnych_zen, pocet_shodnych_muzu, len(y))
pocet_zen = data['sex'].value_counts()[1] #celkový počet žen
pocet_muzu = data['sex'].value_counts()[0] #celkový počet mužů   
print("celkový počet mužů je:", pocet_muzu)
print("celkový počet žen je:", pocet_zen)
print("\nPočet správně určených žen:", pocet_shodnych_zen)
print("\nPočet správně určených mužů:", pocet_shodnych_muzu)
print("\nPřesnost pro sloupce je:", presnost)
####################################
#                                  #
#             Marlow               #
#                                  #
####################################
data = df[['sex','f10', 'f11']]
data = data.dropna(subset=['sex', 'f10', 'f11'], how='any')  
y = data['sex'].values
X = data[['f10', 'f11']].values  
koeficienty = [-22.718, 0.247, 0.178]
X_pred = np.column_stack([np.ones(len(X)), X])
predikce = dfu.rovnice(X_pred, koeficienty)   
binarizovane_predikce = (predikce < -0.004).astype(int) 
shodne_hodnoty = (binarizovane_predikce == y)
pocet_shodnych_zen= np.sum((shodne_hodnoty == True) & (binarizovane_predikce == 1)) 
pocet_shodnych_muzu = np.sum((shodne_hodnoty == True) & (binarizovane_predikce == 0)) 
presnost = dfu.presnost(pocet_shodnych_zen, pocet_shodnych_muzu, len(y))
pocet_zen = data['sex'].value_counts()[1]
pocet_muzu = data['sex'].value_counts()[0]    
print("celkový počet mužů je:", pocet_muzu)
print("celkový počet žen je:", pocet_zen)
print("\nPočet správně určených žen:", pocet_shodnych_zen)
print("\nPočet správně určených mužů:", pocet_shodnych_muzu)
print("\nPřesnost pro sloupce je:", presnost)
####################################
#                                  #
#         Marlow - lebka           #
#                                  #
####################################
df = pd.read_excel("data_git.xlsx", sheet_name="skull final") 
df = df.replace({'M': 0, 'F': 1, 'M?': 0, 'F?':1})
sloupce = df.columns[4:] 
data = df[['sex','f_s1', 'f_s7']]
data = data.dropna(subset=['sex', 'f_s1', 'f_s7'], how='any') 
y = data['sex'].values
X = data[['f_s1', 'f_s7']].values  
koeficienty = [-38.163, 0.108, 0.146]
X_pred = np.column_stack([np.ones(len(X)), X])
predikce= dfu.rovnice(X_pred, koeficienty)   
binarizovane_predikce = (predikce < -0.005).astype(int) 
shodne_hodnoty = (binarizovane_predikce == y)
pocet_shodnych_zen = np.sum((shodne_hodnoty == True) & (binarizovane_predikce == 1)) 
pocet_shodnych_muzu = np.sum((shodne_hodnoty== True) & (binarizovane_predikce == 0)) 
presnost = dfu.presnost(pocet_shodnych_zen, pocet_shodnych_muzu, len(y))
pocet_zen = data['sex'].value_counts()[1]
pocet_muzu = data['sex'].value_counts()[0]    
print("celkový počet mužů je:", pocet_muzu)
print("celkový počet žen je:", pocet_zen)
print("\nPočet správně určených žen:", pocet_shodnych_zen)
print("\nPočet správně určených mužů:", pocet_shodnych_muzu)
print("\nPřesnost pro sloupce je:", presnost)



