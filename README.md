# Logs2Graphs

Referencyjna implementacja [Logs2Graphs](https://github.com/ZhongLIFR/Logs2Graph) do wykrywania anomalii w procesie rozruchu systemu operacyjnego.

To repozytorium zawiera referencyjną implementację metody [Logs2Graphs](https://github.com/ZhongLIFR/Logs2Graph), stworzoną na potrzeby pracy inżynierskiej pt. "Wykrywanie anomalii w procesie rozruchu systemu operacyjnego z wykorzystaniem uczenia maszynowego"

## Środowisko

- Python 3.10.14
- libs.txt zawiera pełną listę zależności wymaganych do uruchomienia kodu.

## Zbiory danych

[Link do pobrania zbiorów danych Linux i Windows](https://drive.google.com/drive/folders/1mv--B6TKiTtcy20SNKtbi1xMEW-IuRoE?usp=sharing)

Po pobraniu, odpowiedni plik należy umieścić w katalogu /Data/Windows/Windows.log i analogicznie /Data/Linux/Linux.log

## Słowniki GloVe

[Link do pobrania słowników GloVe](https://nlp.stanford.edu/data/glove.6B.zip)

Po rozpakowaniu archiwum należy umieścić pliki glove.6B.50d.txt oraz glove.6B.200d.txt w katalogu /Data/Gloves/

## Uruchomienie

Należy zmienić zmienną `root_path` na początku plików python: `GraphGeneration.py`, `Main.py`, `Parser.py`, `Results.ipynb` tak, aby wskazywała odpowiednią ścieżkę do katalogu.

W celu przeprowadzenia eksperymentu i uzyskania wyników dla konkretnego zbioru danych, należy ustawić odpowiednią wartość zmiennej `dataset_name` w pliku `Results.ipynb` i uruchomić wszystkie komórki notatnika.

```python
dataset_name = 'Windows'
dataset_name = 'Linux'
```

## Optymalizacja hiperparametrów

Aby przeprowadzić optymalizację hiperparametrów, należy należy ustawić odpowiednią wartość zmiennej `dataset_name` w pliku `HPO.ipynb` i uruchomić wszystkie komórki notatnika.

```python
dataset_name = 'Windows'
dataset_name = 'Linux'
```

