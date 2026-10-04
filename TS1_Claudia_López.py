import numpy as np
# import matplotlib.pyplot as plt
import math
from TS0_Claudia_López import mi_funcion_sen
from scipy import signal

N = 1000                 # número de muestras
fs = 1000                # frecuencia de muestreo
ts = 1/fs
vmax = math.sqrt(2)      # amplitud
dc = 0                   # desplazamiento vertical
ff = fs/N                # frecuencia variable
ph = 0                   # fase
nn = N

tt = np.arange(0, nn) * ts
xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)

# señal 1: sinusoidal de 2KHz con 10 puntos por periodo
f_sig_1 = 2000
fs_1 = f_sig_1 * 10      # frecuencia de la señal (20kHz * 10 puntos por periodo)
ts_1 = 1/fs_1
tt1, xx1 = mi_funcion_sen(vmax=1, dc=0, ff=f_sig_1, ph=0, nn=N, fs=fs_1)

# señal 2: sinusoidal con potencia máxima de 2W
vmax_2 = math.sqrt(2*2)
tt2, xx2 = mi_funcion_sen(vmax=vmax_2, dc=0, ff=f_sig_1, ph=np.pi/2, nn=N, fs=fs_1)

# señal 3: secuencia aleatoria de ruido (distribuición normal)
px = np.var(xx)            # varianza/potencia de la señal limpia
snr_db = 10
snr_lineal = 10 ** (snr_db/10)
P_ruido = px / snr_lineal
sigma = np.sqrt(P_ruido)

rng = np.random.default_rng()
ruido = rng.normal(0, sigma, size=xx.shape)

xx_ruidosa_normal  = xx + ruido

# señal 4: secuencia aleatoria de ruido (distribuición uniforme)
a = np.sqrt(3 * P_ruido)   # rango de la distribuición uniforme (varianza = a^2 / 3)
ruido_uniforme = rng.uniform(-a, a, size=xx.shape)
xx_ruidosa_uniforme = xx + ruido_uniforme

# señal 5: pulso rectangular de 2KHz, 1W y 50% de ciclo de actividad
xx5 = signal.square(2 * np.pi * f_sig_1 * tt, duty=0.5)

## visualizar el módulo de la transformada de Fourier
X1 = np.fft.fft(xx1)
X2 = np.fft.fft(xx2)
X3 = np.fft.fft(xx_ruidosa_normal)
X4 = np.fft.fft(xx_ruidosa_uniforme)
X5 = np.fft.fft(xx5)

# ¿graficar?
