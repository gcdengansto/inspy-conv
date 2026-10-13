from inspy.energy import Energy

from inspy.crystal import Sample
from inspy.instrument import get_tau, TripleAxisSpectr
from inspy.instrument import Mono
from inspy.instrument import Ana

import numpy as np




mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1)
ana = Ana(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1, thickness =0.2, horifoc=-1)
sample= Sample(3,4,5,90,90,90)

instr=TripleAxisSpectr(efixed=14.7, mono=mono, sample=sample, ana=ana)

hkle=np.array([[2, 0, 0, 0],[2, 0, 0, 2],[2, 0, 0, 2]])

print(hkle)

test=instr.CalcResMatHKL(hkle.T)

print(test)