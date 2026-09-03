# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:04:00 2026

@author: claud
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# SNR|dB = 20dB
# senoidal con a = sqrt(2)
# Px = 1W
# np.var(x)
# np.sqrt(2)

fs = 1000   # frecuencia de muestreo (Hz)
N = 1000
ts =1/fs  

# def: generar senoidal
def mi_funcion_sen(vmax= math.sqrt(2), dc=0, ff=100, ph=0, nn=N, fs=fs):
   
    ts = 1 / fs                     # tiempo entre cada muestra
    tt = np.arange(0, nn) * ts      # tiempo de cada muestra
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return(tt, xx)

vmax = math.sqrt(2)      # amplitud
dc = 0                   # desplazamiento vertical
ff = 3                 # frecuencia variable
ph = 0                   # fase
nn = N

tt = np.arange(0, nn) * ts
xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)

px = np.var(xx)
print(px)

# aquí comento grafica sin ruido
#tt, xx = mi_funcion_sen(vmax = math.sqrt(2))
#plt.plot(tt, xx)
#plt.xlabel('Tiempo[seg]')
#plt.ylabel('Amplitud [V]')
#plt.show()

snr_db = 10  # evaluar diferentes niveles de SNR
snr_lineal = 10 ** (snr_db / 10)  # convertir dB a escala lineal (10log)

# potencia para ruido
P_ruido = px / snr_lineal
sigma = np.sqrt(P_ruido)  # sd del ruido

# secuencia de ruido gaussiano (random)
rng = np.random.default_rng()
ruido = rng.normal(0, sigma, size=xx.shape)

senal_ruidosa = xx  + ruido  # señal + ruido

# graficar
plt.figure(figsize=(10, 6))
plt.plot(tt, xx, label="Señal Original (1W)", linewidth=2)
plt.plot( tt, senal_ruidosa, label=f"Señal Ruidosa (SNR = {snr_db} dB)", alpha=0.7)
plt.title("Estudio de SNR: Señal Senoidal + Ruido Aleatorio Gaussiano")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# ejercicio: graficar separando modulo y fase
# transformada de fourier
X = np.fft.fft(senal_ruidosa)
freqs = np.fft.fftfreq(N, d=ts)

# division entera de N
N2 = N // 2

# el módulo es de 0 a N//2
# (frecuencias positivas)
modulo = np.abs(X[:N2])  
freqs_mod = freqs[:N2]

# la fase es de N//2 hasta N 
# (frecuencias negativas)
fase = np.angle(X[N2:])
freqs_fase = freqs[N2:]

# gráfica módulo
plt.figure(figsize=(10, 4))
plt.plot(freqs_mod, modulo, color='blue', linewidth=1.5)
plt.title("Módulo de la FFT (índices 0 a N/2)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()

# gráfica fase
plt.figure(figsize=(10, 4))
plt.plot(freqs_fase,fase, color='red', linewidth=1.5)
plt.title("Fase de la FFT (índices N/2 a N)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [grados]")
plt.grid(True)
plt.show()

# ejercicio 2: graficar módulo y fase de la señal ORIGINAL
# transformada de fourier
X = np.fft.fft(xx)
freqs_og = np.fft.fftfreq(N, d=ts)

# módulo
modulo_og = np.abs(X[:N2])  
freqs_mod_og = freqs[:N2]

# fase
fase_og = np.angle(X[N2:])
freqs_fase_og = freqs[N2:]

# gráfica módulo señal original
plt.figure(figsize=(10, 4))
plt.plot(freqs_mod_og, modulo_og, color='blue', linewidth=1.5)
plt.title("Módulo de la FFT (índices 0 a N/2)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()

# gráfica fase señal original
plt.plot(freqs_fase_og,fase_og, color='red', linewidth=1.5)
plt.title("Fase de la FFT (índices N/2 a N)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [grados]")
plt.grid(True)
plt.show()

# ejercicio 3: cuantizacion
# cantidad de bits B, rango analogico Vfs
# señal entre +Vfs y -Vfs llevarlo a 256 valores distintos

# parámetros
B = 8           # 8 bits para 256 niveles
Vfs = 2.0       # rango analógico

qq = (2*Vfs)/(2**B)

xx_q = np.round(senal_ruidosa / qq) * qq
nq = xx_q - senal_ruidosa
