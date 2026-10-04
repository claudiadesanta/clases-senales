import numpy as np
import matplotlib.pyplot as plt
import math

N = 1000      # número de muestras
fs = 1000     # frecuencia

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    
    ts = 1 / fs                 # tiempo entre cada muestra
    tt = np.arange(nn) * ts     # tiempo de cada muestra
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    
    return tt, xx

vmax = math.sqrt(2)      # amplitud
dc = 0                   # desplazamiento vertical
ff = fs/N                # frecuencia variable
ph = 0                   # fase
nn = N

tt, xx = mi_funcion_sen(vmax = math.sqrt(2))

# px = np.var(xx)
# print(f"Varianza (o potencia): {px}")

#### GRÁFICA
plt.figure(figsize=(9, 4))
plt.plot(tt, xx, color='b')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo[s]')
plt.ylabel('Amplitud[V]')
plt.tight_layout()
plt.show()