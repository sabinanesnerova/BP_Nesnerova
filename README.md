# BP_Nesnerova
Kód v jazyce Python, díky kterému byly získány výsledky prezentované v bakalářské práci. Zároveň lze zde nalézt kompletní tabulky získaných výsledků a modifikovaná vstupní data
Vstupní antropologická data (data_git) byla modifikována pro jejich anonymitu následujícím způsobem.
- počet jedinců by u obou listů zredukován ze 162 na 77 a ze 113 na 54
- kódy jedinců byly změněny na označení person1, person2,...
- bylo vybráno pouze pár příznaků s výbornými a pár příznaků se špatnými výsledky (u kostry 18, u lebky 15) a ty byly přejmenovány na f_1, f_2,..., u lebky f_s1, f_s2,...
- s těmito daty pak pracují skripty zde vložené

  
- Skripty logistické regrese jsou aplikovány na modifikovaná data, avšak výstupní excelové tabulky přísluší reálným výsledkům antropologických dat.
- aplikace_studii_na_egypt_dat - aplikuje známé studie na modifikovaná antropologická data
- definice_funkci - zde jsou definovány veškeré funkce a statistiky využité v ostatních skriptech
- dva_sloupce - aplikuje vždy logistickou regresi na dva zadané příznaky 
- linearni_regrese_bak - ilustrace lineární regrese 
- log_regrese_1_nezavisla - aplikuje logistickou regresi na každý příznak ze zadaného listu
- odlehle_body - zpracovává odlehlé body ke skriptu linearni_regrese_bak 
- outlier - ilustrace rozdílů mezi tři druhy odlehlých bodů
- postcr_vysledky_1rozmer - kompletní tabulka výsledků pro aplikaci jednorozměrné logistické regrese na kompletní antropologická data - kostra
- postcran_skull_vysledky - kompletní tabulka výsledků pro aplikaci dvojrozměrné logistické regrese na kompletní antropologická data - kostra i lebka
- postcran_vysledky_2rozmer - kompletní tabulka výsledků pro aplikaci dvojrozměrné logistické regrese na kompletní antropologická data - kostra
- postcranial - aplikace logistické regrese na každé dva příznaky - kostra
- postcranial_1priznak - aplikace logistické regrese na každý jeden příznak - kostra
- predikce - skript lineární regrese predikující hodnoty a zbylé statistiky týkající se závislé proěnné na základě nezávislé proměnné zadané uživatelem
- skull - aplikace logistické regrese na každé dva příznaky - lebka
- skull_vysledky_2rozmer - kompletní tabulka výsledků pro aplikaci dvojrozměrné logistické regrese na kompletní antropologická data - lebka
- skull_vysledky_1ozmer - kompletní tabulka výsledků pro aplikaci jednorozměrné logistické regrese na kompletní antropologická data - lebka
- zavislost_vahy_na_vysce - vstupní nemodifikovaná data pro skript lineární regrese
