import numpy as np
from scipy.stats import t,norm
from scipy import stats
import statsmodels.api as sm
K = 3 #konstanta pro hledání odlehlých bodů, nejčastěji 3 nebo 2
L = 1.96 # hodnota kvantilu normálního rozdělení pro alfa=0.05
alfa = 0.05 #nejčastěji volený kvantil, lze měnit
######################################################################
#                                                                    #
#                      LINEÁRNÍ REGRESE                              #
#                                                                    #
######################################################################
def sum_promenna_mocnina(promenna,mocnina):
    #suma proměnné na určitou mpocninu
    sum_promenna_mocnina = np.sum(promenna**mocnina)
    return sum_promenna_mocnina

def reseni_linearni_regrese(matice_nezavislych, zavisla):
    #nalezení regresních koeficientů lineární regrese
    b = np.linalg.inv(matice_nezavislych.T @ matice_nezavislych) @ matice_nezavislych.T @ zavisla
    return b

def rovnice(matice_nezavislych, koeficienty):
    #vypíše rovnici pro model
    rovnice =  matice_nezavislych @ koeficienty
    return rovnice

def prumer(promenna):
    #tato funkce počítá průměr dané proměnné
    prumer = (1/len(promenna))*np.sum(promenna)
    return prumer

def ctverec(promenna):
    #počítá součet druhé mocniny rozdílu jednotlivých hodnot jejich průměru
    rozdil = promenna - prumer(promenna)
    ctverec = sum(rozdil**2)
    return ctverec

def reziduum(matice_nezavislych,zavisla, koeficienty):
    #počítá rozdíly naměřených a spočítaných hodnot
    reziduum = zavisla - matice_nezavislych@koeficienty
    return reziduum

def durbin_watson(matice_nezavislych,zavisla, koeficienty):
    #statistika pro test nezávislosti reziduí
    reziduum_ = reziduum(matice_nezavislych, zavisla, koeficienty)
    rozdil_rezidui = [reziduum_[i+1] - reziduum_[i] for i in range(2, len(reziduum_)-1)] #rozdíl rezidua a rezidua s menším indexem
    druha_mocnina = [rozdil ** 2 for rozdil in rozdil_rezidui]
    reziduum_na2 = [rezid ** 2 for rezid in reziduum_]
    durbin_watson = sum(druha_mocnina) / sum(reziduum_na2)
    return durbin_watson

def vif(matice_nezavislych):
    #koeficient určující multikolinearitu
    vif = []
    for i in range(matice_nezavislych.shape[1]): #získá se koeficient multikolinearity pro vymazání jednotlivých nezávislých
        matice_nezavislych_i = matice_nezavislych[:, i] #vezme i-tý sloupec, pro vyšetření jeho závislosti
        matice_nezavislych_zb = np.delete(matice_nezavislych, i, axis=1) #zbylá matice bez i-tého sloupce
        koeficienty = reseni_linearni_regrese(matice_nezavislych_zb, matice_nezavislych_i) #model bez i-té nezávislé
        R2_i = regresni_koeficient(matice_nezavislych_zb, matice_nezavislych_i, koeficienty)
        vif_i = 1 / (1 - R2_i)
        vif.append(vif_i)
    return vif

def studentizovane_reziduum(matice_nezavislych,zavisla, koeficienty):
    #pro identifikaci outlierů
    rezidua = reziduum(matice_nezavislych, zavisla, koeficienty)
    variace_rez = np.var(rezidua)
    odmocnina = np.sqrt(variace_rez)
    student_rez = rezidua/odmocnina
    return(student_rez)

def rezidualni_rozptyl(matice_nezavislych,zavisla, koeficienty):
    #tato funkce počítá reziduální rozptyl na základě reziduí 
    reziduum_ = reziduum(matice_nezavislych,zavisla, koeficienty)
    rezidualni_rozptyl=(np.sum(reziduum_.T@reziduum_))/(len(zavisla)-len(koeficienty)) # jmenovatel vypovídá o celkové variabilitě závislé proměnné
    return rezidualni_rozptyl

def outlier(matice_nezavislych,zavisla, koeficienty):
    #detekce outlieru
    student_rezi = studentizovane_reziduum(matice_nezavislych,zavisla, koeficienty)
    podminka = abs(student_rezi) > K #podmínka pro nalezení outliera
    outlier = podminka
    return outlier

def matice_pro_outliery(matice_nezavislych):
    inverze = np.linalg.inv(matice_nezavislych.T@matice_nezavislych)
    H = matice_nezavislych@inverze@matice_nezavislych.T
    return H

def leverage_point(matice_nezavislych):
    #identifikace leverage pointu
    H = matice_pro_outliery(matice_nezavislych)
    diagonala = np.diag(H)
    leverage_point = diagonala > (K*sum(diagonala))/len(matice_nezavislych[:,0])
    return leverage_point

def cook_vzd(matice_nezavislych, zavisla, koeficienty):
    #identifikace influentního bodu
    cook_vzdalenost =[]
    s =rezidualni_rozptyl(matice_nezavislych, zavisla, koeficienty)
    zavisla_pred = rovnice(matice_nezavislych, koeficienty) #predikovaná závislá
    for i in range(matice_nezavislych.shape[0]):
       matice_nezavislych_i = np.delete(matice_nezavislych, i, axis=0) #vymaže i-tou nezávislou proměnnou
       zavisla_i = np.delete(zavisla, i)
       koeficienty_i = reseni_linearni_regrese(matice_nezavislych_i, zavisla_i) #model bez i-té nezávislé
       zavisla_pred_i = rovnice(matice_nezavislych_i, koeficienty_i) #predikovaná závislá na základě modelu pro data bez i-té nezávislé
       zavisla_pred_bez_i = np.delete(zavisla_pred, i) #predikce závislé s vymazáním i-té predikce
       D_i = (np.sum((zavisla_pred_bez_i - zavisla_pred_i)**2))/ (len(koeficienty) * s)
       cook_vzdalenost.append(D_i)
    return cook_vzdalenost

def regresni_koeficient(matice_nezavislych,zavisla,koeficienty):
    #vypocet koeficientu udávajícího kvalitu modelu
    rezidualni_soucet_ctvercu = sum((reziduum(matice_nezavislych,zavisla,koeficienty))**2)
    regresni_koeficient = 1 - ((rezidualni_soucet_ctvercu)/ctverec(zavisla))
    return regresni_koeficient 
 
def F_test(matice_nezavislych,zavisla,koeficienty):
    #vypočítá F-statistiku pro testování hypotézy, zda-li je model statisticky významný jako celek
    koeficient_determinace = regresni_koeficient(matice_nezavislych, zavisla, koeficienty)
    prvni_zlomek = koeficient_determinace/(1-koeficient_determinace)
    druhy_zlomek = (len(zavisla)-len(koeficienty)-1)/(len(koeficienty))
    F_test = prvni_zlomek*druhy_zlomek
    return F_test

def standartni_odchylka_koeficientu(matice_nezavislych, zavisla, koeficienty):
    #odchylka odhadu koeficientu
    soucin_matic = np.linalg.inv(matice_nezavislych.T @ matice_nezavislych)
    standartni_odchylka_koeficientu = np.sqrt(rezidualni_rozptyl(matice_nezavislych, zavisla, koeficienty) * np.diag(soucin_matic))
    return standartni_odchylka_koeficientu
        
def t_test_koeficient(matice_nezavislych, zavisla,koeficienty):
    #tato funkce počítá hodnoty pro testování hypotéz, jestli je beta_i statisticky významné
    standartni_odchylka_koeficientu_ = standartni_odchylka_koeficientu(matice_nezavislych, zavisla, koeficienty)
    t_test_koeficient = np.diag(koeficienty/standartni_odchylka_koeficientu_) #kolikrát je větší daný koeficient než jeho chyba
    return t_test_koeficient

def p_hodnota_koeficient_i(matice_nezavislych, zavisla, koeficienty):
    #výpočet p_hodnoty pro jednotlivé koeficienty -> významnost koeficientů
    distr_funkce = stats.t.cdf(np.abs(t_test_koeficient(matice_nezavislych, zavisla, koeficienty)), len(zavisla)-len(koeficienty)) #výpočet distribuční funkce 
    p_hodnota_koeficient_i= 2 * (1 - distr_funkce )
    return p_hodnota_koeficient_i

def hodnota_student_rozdeleni(stupne_volnosti):
    #určuje alfa-kvantil studentova rozdělení
    hodnota_student_rozdeleni = t.ppf(1-(alfa/2), stupne_volnosti)
    return hodnota_student_rozdeleni

intervaly = []    
def intervaly_koeficientu(matice_nezavislych, zavisla, koeficienty):
    #tato funkce vypíše intervaly pro jednotlivé koeficienty
    for i in range(len(koeficienty)):
       odchylka_i = standartni_odchylka_koeficientu(matice_nezavislych, zavisla, koeficienty) [i]
       leva_strana = koeficienty[i] - hodnota_student_rozdeleni(len(zavisla)-len(koeficienty)) * odchylka_i
       prava_strana = koeficienty[i] + hodnota_student_rozdeleni(len(zavisla)-len(koeficienty)) * odchylka_i
       intervaly.append((leva_strana, prava_strana))
    return intervaly

def odhad_zavisle(nova_matice, koeficienty):
    #tato funkce vypíše konkrétní hodnotu, kterou nabývá proměnná na základě vložené nezávislé proměnné
    odhad_zavisle = rovnice(nova_matice, koeficienty)
    return odhad_zavisle

def odchylka_odhadu_int_spol(matice_nezavislych, zavisla, nova_matice, koeficienty):
    #odchylka odhadu na základě vložené nezávislé 
    matice = np.linalg.inv(np.dot(matice_nezavislych.T, matice_nezavislych))
    soucin = np.dot(np.dot(nova_matice, matice), nova_matice.T)
    chyba = np.sqrt(rezidualni_rozptyl(matice_nezavislych, zavisla, koeficienty))
    odchylka_odhadu = hodnota_student_rozdeleni(len(zavisla)-len(koeficienty)) * chyba * np.sqrt(soucin)
    return odchylka_odhadu

def odchylka_odhadu_int_pred(matice_nezavislych, zavisla, nova_matice, koeficienty):
    #odchylka odhadu pro interval predikce
    matice = np.linalg.inv(np.dot(matice_nezavislych.T, matice_nezavislych))
    soucin = np.dot(np.dot(nova_matice, matice), nova_matice.T)
    chyba = np.sqrt(rezidualni_rozptyl(matice_nezavislych, zavisla, koeficienty))
    odchylka_odhadu = hodnota_student_rozdeleni(len(zavisla)-len(koeficienty)) * chyba * np.sqrt(1+soucin)
    return odchylka_odhadu

def interval_spolehlivosti(matice_nezavislych, zavisla, nova_matice, koeficienty):
    #tato funkce vypíše interval, ve kterém se s 95% pravděpodobností vyskytuje predikovaná proměnná (pravděpodobnost by se dala změnit)
    odchylka = odchylka_odhadu_int_spol(matice_nezavislych, zavisla, nova_matice, koeficienty)
    leva_strana = odhad_zavisle(nova_matice, koeficienty) - odchylka
    prava_strana = odhad_zavisle(nova_matice, koeficienty) +  odchylka
    return (leva_strana, prava_strana)

def interval_predikce(matice_nezavislych, zavisla, nova_matice, koeficienty):
    #interval predikce pro odhadovanou závislou proměnnou
    odchylka = odchylka_odhadu_int_pred(matice_nezavislych, zavisla, nova_matice, koeficienty)
    leva_strana = odhad_zavisle(nova_matice, koeficienty) - odchylka
    prava_strana = odhad_zavisle(nova_matice, koeficienty) +  odchylka
    return (leva_strana, prava_strana)
######################################################################
#                                                                    #
#                      LOGISTICKÁ REGRESE                            #
#                                                                    #
######################################################################
def model_logit(zavisla,nezavisle):
    #získání regresních koeficientů logistické regrese
    model = sm.Logit(zavisla, sm.add_constant(nezavisle))
    result = model.fit(disp=False)
    koeficienty = result.params
    shrnuti = result.summary()
    return koeficienty, shrnuti

def y_model(matice_nezavislych, koeficienty): 
    # = pravděpodobnost, že v logistickém modelu získáme binární 1
    y_model = 1/(1+np.exp(-(np.dot(matice_nezavislych,koeficienty))))
    return y_model

def rezidua_log(matice_nezavislych,zavisle, koeficienty):
    #rezidua pro logistický model
    rezidua_log = zavisle - y_model(matice_nezavislych, koeficienty)
    return rezidua_log

def matice_pro_outliery_log(matice_nezavislych, koeficienty):
   matice_D =np.diag( y_model(matice_nezavislych, koeficienty) * (1 - y_model(matice_nezavislych, koeficienty)))
   inverze = np.linalg.inv(np.dot(matice_nezavislych.T, np.dot(matice_D,matice_nezavislych)))
   vnitrni_soucin = np.dot(matice_nezavislych, np.dot(inverze, matice_nezavislych.T))
   H = np.dot(matice_D**(1/2),np.dot(vnitrni_soucin,matice_D**(1/2)))
   return H

def pearson_rezidua(matice_nezavislych, zavisla,koeficienty):
    #ekvivalence studentizovaných reziduí v lineárním modelu
    rezidua = rezidua_log(matice_nezavislych, zavisla, koeficienty)
    odmocnina = np.sqrt((1-np.diag(matice_pro_outliery_log(matice_nezavislych, koeficienty)))*y_model(matice_nezavislych, koeficienty)*(1-y_model(matice_nezavislych, koeficienty)))
    pearson_rezidua = rezidua/odmocnina
    return pearson_rezidua

def outlier_log(matice_nezavislych, zavisla,koeficienty):
    #detekce outliera na základě pearsonových reziduí
    pearson_rez = pearson_rezidua(matice_nezavislych, zavisla, koeficienty)
    podminka = abs(pearson_rez) > K #podmínka pro nalezení outliera
    outlier_log = podminka
    return outlier_log

def log_likelihood(matice_nezavislych, zavisle, koeficienty):
    #logaritmická věrohodnostní funkce modelu
    pred = y_model(matice_nezavislych, koeficienty)
    log_likelihood = np.sum(zavisle * np.log(pred) + (1 - zavisle) * np.log(1 - pred))
    return log_likelihood

def log_likelihood_0(zavisle):
    #logaritmická věrohodnostní funkce modelu pouze s konstantním členem
    p = prumer(zavisle) #při modelu pouze s konstantním členem se bere jako odhad pravděpodobnosti průměr
    log_likelihood_0 = np.sum(zavisle * np.log(p) + (1 - zavisle) * np.log(1 - p))
    return log_likelihood_0

def deviance(matice_nezavislych, zavisle, koeficienty):
    #určuje kvalitu modelu
    deviance = -2*log_likelihood(matice_nezavislych, zavisle, koeficienty)
    return deviance

def mcFadden(matice_nezavislych, zavisle, koeficienty):
    #McFaddenův koeficient kvality logistického modelu
    mcFadden = 1-(log_likelihood(matice_nezavislych, zavisle, koeficienty)/log_likelihood_0(zavisle))
    return mcFadden

def snell(matice_nezavislych, zavisle, koeficienty):
     #snellův koeficient pro výpočet Nagelkerkeho koeficientu
     snell = 1 - np.exp((2*log_likelihood_0(zavisle)-2*log_likelihood(matice_nezavislych, zavisle, koeficienty))/len(zavisle))
     return snell

def nagelkerke(matice_nezavislych, zavisle, koeficienty):
    #Nagelkerkeho koeficient kvality logistického modelu
     snell_ = snell(matice_nezavislych, zavisle, koeficienty)
     nagelkerke = snell_ / (1 - np.exp(2*log_likelihood_0(zavisle) / len(zavisle)))
     return nagelkerke

def lrt(matice_nezavislych, zavisle, koeficienty):
    #věrohodnostní poměr pro testování celkové statistické významnosti logistického modelu
    log_likehood = log_likelihood(matice_nezavislych, zavisle, koeficienty)
    log_likehood_0 = log_likelihood_0(zavisle)
    lrt = -2 * (log_likehood_0 - log_likehood) #následné porovnání s alfa-kvantilem chí kvadrát rozdělení
    return lrt

def fisherova_matice(matice_nezavislych, predikovana_zavisla):
    #informační matice
    soucin_diag = np.diag(predikovana_zavisla * (1 - predikovana_zavisla))
    fisherova_matice = np.dot(np.dot(matice_nezavislych.T, soucin_diag), matice_nezavislych)
    return fisherova_matice

def odchylka_log(matice_nezavislych,predikovana_zavisla):
    #odchylka pro jednotlivé koeficienty
    fish_matice = fisherova_matice(matice_nezavislych, predikovana_zavisla)
    odchylka_log = np.sqrt(np.diag(np.linalg.inv(fish_matice)))
    return odchylka_log

def walduv_test(matice_nezavislych, predikovana_zavisla, koeficienty):
    #pro testování hypotézy, zda-li je koeficient beta_i statisticky významný
    W_statistika =koeficienty**2/odchylka_log(matice_nezavislych,predikovana_zavisla)**2
    W = np.sqrt(W_statistika) #následné porovnání s alfa-kvantilem normálního rozdělení
    return W

def interval_spolehlivosti_log(matice_nezavislych, predikovana_zavisla, koeficienty):
    #intervaly, ve kterých se s 95% pravděpodobností nachází dané koeficienty logistické regrese
    leva_strana=koeficienty - L*odchylka_log(matice_nezavislych, predikovana_zavisla) 
    prava_strana = koeficienty + L*odchylka_log(matice_nezavislych, predikovana_zavisla)
    return (leva_strana, prava_strana)

def p_hodnota_log(matice_nezavislych, predikovana_zavisla, koeficienty):
    #výpočet p_hodnoty pro jednotlivé koeficienty -> významnost koeficientů
    distr_funkce = stats.norm.cdf(np.abs(walduv_test(matice_nezavislych, predikovana_zavisla, koeficienty))) 
    p_hodnota_log= 2 * (1 - distr_funkce )
    return p_hodnota_log

def presnost(spravne_zeny, spravne_muzi, celkovy_pocet):
    #přesnost modelu
    presnost = (spravne_zeny + spravne_muzi)/celkovy_pocet
    return presnost

def precission(spravne_zeny, spatne_zeny):
    precision = (spravne_zeny)/ (spravne_zeny+ spatne_zeny)
    return precision

def recall(spravne_zeny,spatne_muzi):
    recall = (spravne_zeny)/(spravne_zeny+spatne_muzi)
    return recall

def f1_skore(spravne_zeny, spatne_zeny, spatne_muzi):
    #f-1 skóre modelu (určuje kvalitu)
    f1_skore = (2*precission(spravne_zeny, spatne_zeny)*recall(spravne_zeny,spatne_muzi))/(precission(spravne_zeny, spatne_zeny)+recall(spravne_zeny,spatne_muzi))
    return f1_skore

def leverage_point_log(matice_nezavislych, koeficienty):
    #identifikace leverage pointu
    H = matice_pro_outliery_log(matice_nezavislych, koeficienty)
    diagonala = np.diag(H)
    leverage_point_log = diagonala > (K*sum(diagonala))/len(matice_nezavislych) #podmínka pro nalezení leverage pointu
    return leverage_point_log

def cook_vzd_log(matice_nezavislych, zavisla, koeficienty):
    #identifikace influentního bodu
    cook_vzdalenost =[]
    n,p = matice_nezavislych.shape
    y_pred = y_model(matice_nezavislych, koeficienty)
    H = matice_pro_outliery_log(matice_nezavislych, koeficienty)
    for i in range(n):
       matice_nezavislych_i = np.delete(matice_nezavislych, i, axis=0) #vymaže vždy i-té pozorování
       zavisla_i = np.delete(zavisla, i)
       koeficienty_i,shrnuti = model_logit(zavisla_i, matice_nezavislych_i[:, 1:]) #koeficienty modelu bez i-tého pozorování
       y_pred_i = y_model(matice_nezavislych_i, koeficienty_i) #pravděpodobnosti pro model bez i-tého pozorování
       h_ii = H[i, i]
       y_pred_bez_i = np.delete(y_pred, i) #pravděpodobnosti modelu se všemi pozorováními s vynecháním i-té pravděpodobnosti
       D_i = (np.sum(y_pred_bez_i - y_pred_i)** 2 * h_ii) / (p * (1 - h_ii) ** 2) 
       cook_vzdalenost.append(D_i)
    return cook_vzdalenost

def procenta_stupnice(promenna, maximum):
    return (promenna / maximum) * 100




