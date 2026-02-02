import time
import sys
# Importamos nuestros módulos locales
from src.model import AMPL_MODEL_CODE, randomInstance, instanceToAmpl
from src.genetic import geneticAlgorithmParallel
from src.utils import download_data, parse_num_cds, plot_convergence

# Configuración
POP_SIZE = 30       
GENERATIONS = 100    
N_CORES = 15       
LICENSE_UUID = "f6447db7-92ab-4268-a78b-70bbb1406031"
DATA_URL = "https://gist.githubusercontent.com/athersoft/c6baed29465f509c315c2f5fa7db93b4/raw/393643d794fba167cfe9ec6a3c5c91c3a8cd1d48/80x40-chatgpt.dat"

if __name__ == "__main__":
    print("--- INICIANDO ALGORITMO GENÉTICO HÍBRIDO ---")
    
    # 1. Obtener Datos
    currentInstance = download_data(DATA_URL)
    if not currentInstance:
        sys.exit(1)
        
    num_cds = parse_num_cds(currentInstance)
    print(f"Instancia cargada con {num_cds} CDs candidatos.")

    # 2. Ejecutar GA
    startTime = time.time()
    
    bestChrom, bestCost, costHistory, timeList = geneticAlgorithmParallel(
        modelCode=AMPL_MODEL_CODE,
        dataStr=currentInstance,
        numCds=num_cds,
        popSize=POP_SIZE,
        generations=GENERATIONS,
        nJobs=N_CORES,
        licenseUuid=LICENSE_UUID
    )
    
    endTime = time.time()
    
    # 3. Resultados
    print(f"\n--- RESULTADOS FINALES ---")
    print(f"Tiempo Total: {endTime - startTime:.2f} s")
    print(f"Mejor Costo: {bestCost:,.2f}")
    
    cdsAbiertos = [i for i, x in enumerate(bestChrom) if x == 1]
    print(f"CDs Abiertos ({len(cdsAbiertos)}): {cdsAbiertos}")

    plot_convergence(costHistory)