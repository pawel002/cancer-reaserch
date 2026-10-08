"""Model referencyjny: anatomia pacjenta z GliODIL plus radioterapia.

GliODIL dostarcza mapy istoty białej i szarej oraz segmentację guza. Na tej
podstawie :mod:`patient` odtwarza ciągłą gęstość komórek i buduje pole wiązki,
:mod:`fk_rt_gpu` całkuje równanie Fishera--Kołmogorowa z dołożonymi członami
radioterapii, a :mod:`cohort` spina to w generator całej kohorty.

Radioterapii nie ma w oryginalnym GliODIL; została dodana na potrzeby tego
badania jako pole uszkodzeń ``Z`` wraz z członem zabijania komórek.
"""
