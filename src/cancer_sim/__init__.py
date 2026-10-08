"""Fizyka + uczenie maszynowe dla zredukowanej dynamiki guza pod radioterapią.

Pakiet zawiera trzy warstwy:

* :mod:`cancer_sim.realdata` --- model referencyjny: anatomia pacjenta z GliODIL,
  pole wiązki i trójwymiarowy solver reakcyjno-dyfuzyjny z radioterapią (GPU).
* :mod:`cancer_sim.gpu` --- modele zastępcze (ODE, NODE, PI-NODE) oraz wsadowe
  dopasowanie tysięcy niezależnych modeli w jednym tensorze.
* :mod:`cancer_sim.viz` --- rysunki i animacje użyte w raporcie.

Punkty wejścia eksperymentów znajdują się w katalogu ``experiments/``.
"""
