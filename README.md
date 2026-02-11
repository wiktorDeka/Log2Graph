# Log2Graph

Referencyjna implementacja Log2Graph do wykrywania anomalii w procesie rozruchu systemu operacyjnego.

To repozytorium zawiera referencyjną implementację metody Log2Graph stworzoną na potrzeby pracy inżynierskiej pt. "Wykrywanie anomalii w procesie rozruchu systemu operacyjnego z wykorzystaniem uczenia maszynowego"

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

W celu przeprowadzenia eksperymentu i uzyskania wyników, należy uruchomić wszystkie komórki notatnika Jupyter: `Results.ipynb`
