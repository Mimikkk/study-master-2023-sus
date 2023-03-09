Zadanie domowe: implementacja własnego drzewa decyzyjnego uczonego ERM do problemu regresji. Zaimplementowany algorytm
nie musi (ale może) być analogiczny do zaprezentowanego na zajęciach algorytmu dla klasyfikacji. Wszelkie przejawy
inwencji twórczej wskazane.

Rozwiązanie należy wgrać na
platformę [Kaggle](https://www.kaggle.com/account/login?returnUrl=%2Ft%2F3f60a54f1c0b4d38a26b88b4df1d024f), a kod wysłać
prowadzącemu do oceny wraz z informacją jaką funkcję oceny
stosuje zaimplementowany algorytm. Co więcej, jeśli są jakieś nietypowe ograniczenia przestrzeni hipotez, również proszę
o informację np. w komentarzu.

**Schemat zadania**:

Ponieważ Twoim zadaniem jest zaprojektowanie algorytmu uczącego, zadanie jest oceniane na kilku zbiorach danych (a nie
tylko na jednym). Wszystkie zbiory danych można pobrać jako plik data.zip, który zawiera następujące pliki:

[id]-X.csv — plik z cechami obserwacji w zbiorze uczącym
[id]-Y.csv — plik ze złotym standardem dla zbioru uczącego
[id]-test.csv — plik z cechami obserwacji zbioru testowego
gdzie [id] to identyfikatory kolejnych zbiorów danych od 1 do 13.

Rezultatem Twojego algorytmu dla danego zbioru danych powinien być plik [id].csv zawierający kolejne wartości predykcji
algorytmu dla pliku [id]-test.csv. Każda predykcja powinna być w osobnej linijce, a plik może zawierać również tytuł
kolumny "Y". Podsumowując: ten plik ma wyglądać tak jak plik [id]-Y.csv tylko że dla zbioru testowego.

Po dokonaniu predykcji dla wszystkich zbiorów danych, pliki 1.csv, 2.csv, …, 13.csv należy połączyć w ostateczny plik
rozwiązania, spełniający wymagania Kaggle. Plik te przygotowujemy skryptem create_sol.py.

> python3 create_sol.py FOLDER_Z_ROZWIĄZANIAMI PLIK_WYJŚCIOWY

**Przykład wywołania**:
> python3 create_sol.py my_solutions/ output.csv

**Schemat oceniania**:

- wynik na ukrytym zbiorze testowym (ewaluacja dostępna po oddaniu zadania domowego)  powyżej baseline 1 +20%

- wynik na ukrytym zbiorze testowym (ewaluacja dostępna po oddaniu zadania domowego)  powyżej baseline 2 +40%

- wynik na ukrytym zbiorze testowym (ewaluacja dostępna po oddaniu zadania domowego)  powyżej baseline 3 +40%

- najlepszy wynik na ukrytym zbiorze testowym +10%
- używanie niedozwolonych bibliotek, skopiowanie kodu lub jego fragmentów z internetu = wyzerowanie wszystkich punktów.

Powyższe punkty się sumują. Za zadanie można więc w związku z tym uzyskać więcej niż 100% (punkty bonusowe)

## Opis rozwiązania

Zostało utworzone [rozwiązanie](./src/main.py), które tworzy drzewo decyzyjne przy wykorzystaniu:

- Metoda podziału na podzbiory o najlepszym [zysku wariacji](./src/mods/utils/math.py) (ang. *Variance Reduction*),
  który jest zdefiniowany jako różnica między wariancją całego zbioru a sumą wariancji podzbiorów po podziale.
- Jako [kryterium zatrzymania podziałów](./src/mods/structures/node.py) podczas treningu została użyta maksymalna głębokość drzewa (ang. *max_depth*)
  oraz minimalna wielkość podziału (ang. *min_samples_per_split*).
- Przy trenowaniu wysłanego rozwiązania zostało użyte 2.5% zbioru całkowitego jako minimalna wielkość podziału, a
  maksymalna głębokość drzewa wynosiła 5.
- Jako kandydatów rozpatrywanych podczas podziału były wszystkie dostępne wartości zaobserwowane dla każdego z kryteriów, co spowodowało długi trening ( 2h dla wszystkich drzew ), ale też dało dobre wyniki.
