"""
Données numériques liées au radar et vecteurs R, v des axes distance, vitesse des graphes Doppler,
(+ dist. entre antennes + nbr. ambiguités)
pour les prises de vue réalisées dans le cadre du mémoire :

Synchronization of Multimodal Data Flows for Real-Time AI Analysis
Juin 2022, LEDENT François
"""


import numpy as np

c = 3e8
B = 554e6
N = 256
M = 4
lbd = 0.012426
Sclk = 12
fclk = 38461538
Nr = 256
delay = 1147

dR = c/(2*B) * (N+M)/N
R = np.arange(N) * dR

# for oversampling
def RN(array_size): return np.linspace(0, (N-1) * dR, array_size)

dv = fclk*lbd / (2*Nr*(Sclk*(N+M) + delay))
v = np.arange(-N//2, N//2, 1) * dv

# for oversampling
def vN(array_size): return np.linspace(-N//2 * dv, (N//2 - 1) * dv, array_size)

DX, DY = 0.022, 0.040

ambigx = lbd / DX
ambigy = lbd / DY
