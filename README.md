# Modelowanie wzrostu guza mózgu pod radioterapią

Porównanie trzech modeli zastępczych przewidujących **skumulowaną masę guza**
pod radioterapią: mechanistycznego **ODE**, czysto uczonej **NODE** oraz
hybrydowego **PI-NODE** (rdzeń fizyczny plus uczony człon domykający).

Dane referencyjne powstają z anatomii 117 rzeczywistych pacjentów z glejakiem
(framework GliODIL), rozszerzonej o radioterapię: pole naprawialnych uszkodzeń
`Z` oraz człon zabijania komórek, których oryginalny GliODIL nie zawiera.
Pełny opis metody, wyników i wniosków: **`report/main.tex`**.

## Pobranie danych

Zbiór GliODIL nie jest częścią repozytorium: zawiera dane obrazowe pacjentów i
waży 1,3 GB. Poniższy skrypt pobiera go i rozpakowuje dokładnie tam, gdzie
szukają go skrypty, czyli do `real_data/data_GliODIL_essential/`.

```bash
#!/usr/bin/env bash
set -euo pipefail

URL=https://huggingface.co/datasets/m1balcerak/GliODIL/resolve/main/data_GliODIL_essential.zip
ZIP=real_data/data_GliODIL_essential.zip

mkdir -p real_data
curl -L -C - -o "$ZIP" "$URL"          # -C - wznawia przerwane pobieranie
unzip -q -o "$ZIP" -d real_data        # archiwum ma juz katalog data_GliODIL_essential/

ls -d real_data/data_GliODIL_essential/data_* | wc -l   # oczekiwane: 152
```

Archiwum zawiera 152 pacjentów, każdy z `segm.nii.gz`, `segm_rec.nii.gz`,
`t1_wm.nii.gz`, `t1_gm.nii.gz` i `t1_csf.nii.gz` (58 ma dodatkowo `FET.nii.gz`).
Kontrola jakości w kroku 1 odrzuca część przypadków, więc do dalszych
eksperymentów wchodzi 117 pacjentów.

Dane pochodzą z pracy Balcerak M., Weidner J., Karnakov P. i in.,
*Individualizing glioma radiotherapy planning by optimization of a data and
physics-informed discrete loss*, Nature Communications 16, 5982 (2025),
[doi:10.1038/s41467-025-60366-4](https://doi.org/10.1038/s41467-025-60366-4),
licencja MIT.

## Co jest w repozytorium

| katalog | zawartość |
|---|---|
| `src/cancer_sim/realdata/` | model referencyjny 3-D: geometria pacjenta, pole wiązki, solver na GPU |
| `src/cancer_sim/gpu/` | modele zastępcze, wsadowe dopasowanie tysięcy modeli w jednym tensorze |
| `src/cancer_sim/viz/` | rysunki i animacje użyte w raporcie |
| `experiments/` | skrypty odtwarzające wszystkie wyniki |
| `report/` | raport w LaTeX-u wraz z rysunkami |
| `results/` | podsumowania liczbowe (duże pliki pośrednie są pomijane w repo) |

## Wymagania

Python 3.13 oraz karta NVIDIA z CUDA. Obliczenia prowadzono na czterech
RTX 6000 Ada (48 GB VRAM), ale skrypty przyjmują dowolną listę kart przez
`--gpus`.

```bash
uv sync
```

## Jak odtworzyć wyniki

Kroki trzeba wykonać w tej kolejności, bo każdy korzysta z wyjścia
poprzedniego. Czasy podano dla czterech kart.

```bash
# 1. Dane referencyjne: 3 trajektorie 3-D na pacjenta         (~6 min)
python experiments/gen_cohort.py --gpus 0,1,2,3

# 2. Dopasowanie modeli zastępczych                           (~20 h GPU)
#    pinode  - siatka wag PI-NODE          -> tabela dokładności
#    pinode2 - wagi poniżej siatki, omega=0 -> najlepszy wariant
#    pinode3 - modele odniesienia ODE/NODE  -> tabela dokładności
#    pinode4 - wagi uczone zamiast ustalanych
for g in pinode pinode2 pinode3 pinode4; do
    python experiments/bench.py --group $g --gpus 0,1,2,3 --per-gpu 2
    python experiments/aggregate_bench.py --group $g
done

# 3. Nieusuwalna część błędu                                  (~20 min)
python experiments/closure_gap.py

# 4. Pasma niepewności: wpływ liczby i zaszumienia obserwacji (~2,5 h GPU)
python experiments/band_fits.py --gpus 0,1 --members 80
python experiments/band_report.py          # liczby + rysunek 4

# 5. Animacje i rysunki 1-3 raportu                           (~3 min)
python experiments/make_animations.py

# 6. Wagi uczone zamiast ustalanych: liczby do tabeli
python experiments/trained_weights.py
```

Każdy skrypt uruchomiony z `--help` wypisze swoje opcje. Kroki 3-6 korzystają
z wyników kroku 2, więc same nie zadziałają na pustym repozytorium.

## Raport


Rysunki w `report/src/` są wersjonowane, więc raport składa się bez
powtarzania obliczeń. Kroki 4 i 5 nadpisują je, jeśli przeliczysz eksperymenty.
