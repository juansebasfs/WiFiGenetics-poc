#Otimizacao_WiFi_com_AG_v2
import numpy as np
import pandas as pd
import random

# ============================
# Parámetros del GA
# ============================

POP_SIZE = 100          # Tamanho da população
MAX_GENERATIONS = 1000  # Número máximo de gerações
STAGNATION_LIMIT = 200  # Gerações sem melhoria
MUTATION_RATE = 0.02    # Taxa de mutação
ELITISM = 1             # Número de indivíduos de elite a serem mantidos
CHANNELS = [1, 6, 11]   # Canais disponíveis
# ============================
# Lectura del archivo RSSI
# ============================
# Suponemos un archivo "rssi_matrix.csv" NxN con RSSI[i][j]
rssi_df = pd.read_csv('rssi_matrix.csv', header=None)
RSSI = rssi_df.values
N = RSSI.shape[0]  # Número de APs

# Normalizar la matriz RSSI
RSSI = np.abs(RSSI)  # Tomar valores absolutos
RSSI_normalizado = (RSSI - RSSI.min()) / (RSSI.max() - RSSI.min())

# Fijar la semilla (por ejemplo, 42)
random.seed(15)

# Resto del código del algoritmo genético...


# ============================
# Funciones del GA
# ============================

def generar_individuo():
    # Genera un individuo aleatorio asignando canales al azar a cada AP
    return [random.choice(CHANNELS) for _ in range(N)]

def generar_poblacion(tam_pob):
    return [generar_individuo() for _ in range(tam_pob)]

def calcular_fitness(individuo, RSSI_normalizado):
    # Calcular Icom con valores absolutos de RSSI Normalizado
    # IAP_i = (sum(|RSSI[i][j]| si mismo canal) / N)
    # Icom = (sum IAP_i) / N
    IAP = []
    for i in range(N):
        canal_i = individuo[i]
        sum_abs_rssi_mismo_canal = 0.0
        for j in range(N):
            if i != j and individuo[j] == canal_i:
                sum_abs_rssi_mismo_canal += abs(RSSI_normalizado[i][j])
        IAP.append(sum_abs_rssi_mismo_canal / N)
    Icom = sum(IAP) / N
    # Fitness = 1 / (Icom + epsilon), para maximizar fitness y minimizar Icom
    fitness = 1.0 / (Icom + 1e-9)
    return fitness

def seleccion_torneo(poblacion, fitnesses, k=3):
    # k es el número de individuos que compiten en cada torneo
    # Tomar k individuos aleatoriamente
    indices = random.sample(range(len(poblacion)), k)
    candidatos = [(poblacion[i], fitnesses[i]) for i in indices]
    # El mejor de los k candidatos es el ganador del torneo
    ganador = max(candidatos, key=lambda x: x[1])
    return ganador[0]

def seleccion_torneo(poblacion, fitnesses, k=3):
    # k es el número de individuos que compiten en cada torneo
    # Tomar k individuos aleatoriamente
    indices = random.sample(range(len(poblacion)), k)
    candidatos = [(poblacion[i], fitnesses[i]) for i in indices]
    # El mejor de los k candidatos es el ganador del torneo
    ganador = max(candidatos, key=lambda x: x[1])
    return ganador[0]

def seleccion_ruleta(poblacion, fitnesses):
    # Selección por ruleta
    total_fit = sum(fitnesses)
    if total_fit == 0:
        total_fit = 1e-9
    
    pick = random.uniform(0, total_fit)
    current = 0
    for ind, fit in zip(poblacion, fitnesses):
        current += fit
        if current > pick:
            return ind
    return poblacion[-1]

def crossover_un_punto(padre1, padre2):
    # Cruzamiento de un punto
    punto = random.randint(1, N-1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

def crossover_dos_puntos(padre1, padre2):
    # Suponiendo que N es el largo del cromosoma (cantidad de APs)
    punto1 = random.randint(1, N-2)
    punto2 = random.randint(punto1+1, N-1)

    # El segmento entre punto1 y punto2 del primer hijo proviene del segundo padre
    hijo1 = padre1[:punto1] + padre2[punto1:punto2] + padre1[punto2:]
    # El segmento entre punto1 y punto2 del segundo hijo proviene del primer padre
    hijo2 = padre2[:punto1] + padre1[punto1:punto2] + padre2[punto2:]

    return hijo1, hijo2

def mutacion(individuo):
    # Mutación con probabilidad MUTATION_RATE
    for i in range(N):
        if random.random() < MUTATION_RATE:
            canal_actual = individuo[i]
            nuevos_canales = [c for c in CHANNELS if c != canal_actual]
            individuo[i] = random.choice(nuevos_canales)
    return individuo

# ============================
# Ejecución del GA
# ============================
def ejecutar_ga():
    poblacion = generar_poblacion(POP_SIZE)
    
    # Evaluar fitness inicial
    fitnesses = [calcular_fitness(ind, RSSI_normalizado) for ind in poblacion]
    mejor_fitness = max(fitnesses)
    mejor_individuo = poblacion[np.argmax(fitnesses)]
    generaciones_sin_mejora = 0
    
    for gen in range(MAX_GENERATIONS):
        # Criterio de estancamiento
        if generaciones_sin_mejora >= STAGNATION_LIMIT:
            print("Estagnação alcançada. Terminando na geração:", gen)
            break
        
        nueva_poblacion = []
        
        # Elitismo: conservar los mejores individuos
        poblacion_ordenada = [x for _, x in sorted(zip(fitnesses, poblacion), key=lambda pair: pair[0], reverse=True)]
        for e in range(ELITISM):
            nueva_poblacion.append(poblacion_ordenada[e])
        
        # Generar el resto de la población
        while len(nueva_poblacion) < POP_SIZE:
            #padre1 = seleccion_ruleta(poblacion, fitnesses)
            #padre2 = seleccion_ruleta(poblacion, fitnesses)
            padre1 = seleccion_torneo(poblacion, fitnesses, k=3)
            padre2 = seleccion_torneo(poblacion, fitnesses, k=3)

            #hijo1, hijo2 = crossover_un_punto(padre1, padre2)
            hijo1, hijo2 = crossover_dos_puntos(padre1, padre2)
            
            hijo1 = mutacion(hijo1)
            hijo2 = mutacion(hijo2)
            
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < POP_SIZE:
                nueva_poblacion.append(hijo2)
        
        poblacion = nueva_poblacion
        fitnesses = [calcular_fitness(ind, RSSI_normalizado) for ind in poblacion]
        
        max_fit_gen = max(fitnesses)
        if max_fit_gen > mejor_fitness:
            mejor_fitness = max_fit_gen
            mejor_individuo = poblacion[np.argmax(fitnesses)]
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

         
    # Calcular Icom final del mejor individuo
    # Repetimos el cálculo de Icom para reportarlo:
    IAP = []
    for i in range(N):
        canal_i = mejor_individuo[i]
        sum_abs_rssi_mismo_canal = 0.0
        for j in range(N):
            if i != j and mejor_individuo[j] == canal_i:
                sum_abs_rssi_mismo_canal += abs(RSSI_normalizado[i][j])
        IAP.append(sum_abs_rssi_mismo_canal / N)
    Icom = sum(IAP) / N
    
    return mejor_individuo, Icom

#Calcular Icom sin optimizar, si todos estuvieran en canal 6 
    
def calcular_Icom(individuo, RSSI_normalizado):
    IAP = []
    for i in range(N):
        canal_i = individuo[i]
        sum_abs_rssi_mismo_canal = 0.0
        for j in range(N):
            if i != j and individuo[j] == canal_i:
                sum_abs_rssi_mismo_canal += abs(RSSI_normalizado[i][j])
        IAP.append(sum_abs_rssi_mismo_canal / N)
    Icom = sum(IAP) / N
    return Icom    
    
todos_6 = [ 6 ]*N
icom_todos_6 = calcular_Icom(todos_6, RSSI_normalizado)
print("Icom com todos os APs no canal 6:", icom_todos_6)
    
if __name__ == "__main__":
    mejor_solucion, icom_minimo = ejecutar_ga()
    print("Melhor solução encontrada:", mejor_solucion)
    print("Icom mínimo alcançado:", icom_minimo)
    
    # Guardar la solución en un archivo CSV
    df_sol = pd.DataFrame(mejor_solucion, columns=["Canal"])
    df_sol.to_csv("APs_Canais_Configuracao.csv", index_label="AP")
    print("Arquivo APs_Canais_Configuracao.csv criado com a configuração final.")
    
from itertools import product

def brute_force_optimal(channels, N, RSSI_normalizado):
    """
    Encuentra la mejor configuración mediante fuerza bruta.
    - channels: lista de canales disponibles (ej., [1, 6, 11])
    - N: número de APs
    - RSSI_normalizado: matriz de RSSI normalizada
    """
    best_fitness = 0
    best_config = None

    for config in product(channels, repeat=N):
        fitness = calcular_fitness(config, RSSI_normalizado)  # Usa la misma función
        if fitness > best_fitness:
            best_fitness = fitness
            best_config = config

    # Calcular el Icom para la configuración óptima
    optimal_icom = calcular_Icom(best_config, RSSI_normalizado)
    return best_config, best_fitness, optimal_icom

# Ejemplo de uso
if __name__ == "__main__":
    # Llamamos a la búsqueda por fuerza bruta
    optimal_config, optimal_fitness, optimal_icom = brute_force_optimal(CHANNELS, N, RSSI_normalizado)
    print("Configuración óptima por fuerza bruta:", optimal_config)
    print("Fitness óptimo por fuerza bruta:", optimal_fitness)
    print("Icom óptimo por fuerza bruta:", optimal_icom)
import matplotlib.pyplot as plt

#def plot_fitness_space(channels, N, RSSI_normalizado):
    #"""
    #Genera un gráfico del espacio de búsqueda de fitness.
    #"""
  #  configurations = list(product(channels, repeat=N))
  #  fitness_values = [calcular_fitness(config, RSSI_normalizado) for config in configurations]

   # plt.figure(figsize=(10, 6))
   # plt.plot(range(len(configurations)), fitness_values, 'o')
   # plt.title("Espacio de Búsqueda de Fitness")
   # plt.xlabel("Configuración (índice)")
   # plt.ylabel("Fitness")
   # plt.show()

# Ejemplo de uso
#if __name__ == "__main__":
#    plot_fitness_space(CHANNELS, N, RSSI_normalizado)