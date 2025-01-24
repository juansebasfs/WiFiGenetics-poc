#Criar_arquivo_RSSI-10APs
import numpy as np
import pandas as pd

N = 10
# Generar valores RSSI aleatorios entre -90 y -30 para i != j
# y 0 para la diagonal (AP con sí mismo)
RSSI = np.random.uniform(-90, -30, size=(N, N))

# Poner 0 en la diagonal
for i in range(N):
    RSSI[i][i] = 0.0

# Crear un DataFrame y guardarlo en CSV
df = pd.DataFrame(RSSI)
df.to_csv("rssi_matrix.csv", header=False, index=False)

print("Matriz RSSI generada:")
print(df)
print("Archivo rssi_matrix.csv creado exitosamente.")
