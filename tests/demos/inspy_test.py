# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:54:46 2020

@author: gcdeng
"""

import numpy as np
from inspy.crystal import Sample
from inspy.instrument import get_tau, Mono, Ana,TripleAxisSpectr


print(get_tau('PG(002)'))

mono = Mono(tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1)

ana  = Ana( tau='PG(002)', mosaic=30, vmosaic=30, height=20, width=20, depth=0.2, rh=150, rv=150, direct=-1, thickness =0.2, horifoc=1)
sample= Sample(4,5,6,90,90,90, u=[1,0,0], v=[0,0,1])
print(mono)
print(ana)
print(sample)
#print(sample.u)
#print(sample.v)


instr=TripleAxisSpectr(efixed=14.7, mono=mono, sample=sample, ana=ana)
instr.moncor=1
instr.method=1

hkle=np.array([[1.5, 0, 0, 0],[1.5, 0, 0, 1],[1.5, 0, 0, 2],[1.5, 0, 0, 3],[1.5, 0, 0, 4]]).T

test=instr.CalcResMatHKL(hkle)

print(instr.R0)
print(instr.RMS)
print(instr.RM)
instr.ResolutionPlot(hkle)
