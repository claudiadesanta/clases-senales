import numpy as np
import matplotlib.pyplot as plt
import math

fs = 200              # frecuencia de muestreo (Hz)
N = 1000
ts =1/fs 
k_0 = N/4              # desplazamiento vertical
ff = (fs/N)              # frecuencia variable (fs/N es delta_f)
# Vmax = np.sqrt(2)          # amplitud (potencia normalizada = 1) - definida en la propia función

desintonias = [0.0, 0.25, 0.5]
# def: generar senoidal
def mi_funcion_sen(vmax=math.sqrt(2), dc=0, ff=ff, ph=0, nn=N, fs=fs):
   
    ts = 1 / fs                     # tiempo entre cada muestra
    tt = np.arange(0, nn) * ts      # tiempo de cada muestra
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return(tt, xx)

N2 = N // 2
k_eje = np.arange(N2)


N_zp = (10 * N) // 2
k_eje_zp = np.arange(N_zp) / 10

plt.figure(figsize=(10, 5))

for delta_k in desintonias:
    k_actual = k_0 + delta_k
    frecuencia = k_actual + ff
    tt, xx = mi_funcion_sen(vmax = math.sqrt(2), ff=frecuencia, nn=N, fs=fs)
    # calculamos DFT
    X = np.fft.fft(xx)
    # densidades espectrales de potencia
    # 1. modulo (con ello multiplicar por dos)
    modulo = np.abs(X[:N2])/N
    amplitud = modulo*2
    # obtenemos amplitud y queremos densidad de POTENCIA
    # 2. para ello elevamos al cuadrado
    densidad_potencia = modulo**2
    plt.plot(k_eje, densidad_potencia, marker='o', markersize=2.5, linestyle='--', linewidth=0.5)
    
    potencia_unitaria = np.mean(xx **2)
    densidad_potencia_parseval = (np.abs(X)/N) **2
    potencia_frecuencia = np.sum(densidad_potencia_parseval)
    print(f"Desintonía {delta_k} -> Potencia unitaria: {potencia_unitaria} W")
    
    xx_zp = np.pad(xx, (0, 9 * N), mode='constant')
    
    X_zp = np.fft.fft(xx_zp)
    modulo_zp = np.abs(X_zp[:N_zp])/N
    densidad_potencia_zp = modulo_zp**2
    plt.plot(k_eje_zp, densidad_potencia_zp, label=f'Desintonía = {delta_k}', linewidth=1)

# gráfica
plt.xlim([k_0 - 15, k_0 + 15])    
plt.plot(tt, xx)
plt.xlabel('Tiempo[seg]')
plt.ylabel('Amplitud [V]')
plt.show()
