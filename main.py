import time
import sys
from src.model import AMPL_MODEL_CODE, randomInstance, instanceToAmpl
from src.genetic import geneticAlgorithm
from src.utils import download_data, parse_num_cds, plot_convergence

#Instancias
#Generadas por diferentes modelos LLM 

instancesChatGPT = [
    "https://gist.githubusercontent.com/athersoft/c6baed29465f509c315c2f5fa7db93b4/raw/393643d794fba167cfe9ec6a3c5c91c3a8cd1d48/80x40-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/60304f8af5b3dfc33cf62094a0cc78d6/raw/0a1eb0599c84622853b9294abe337df32b60df7a/100x50-chatgpt",
    "https://gist.githubusercontent.com/athersoft/6bd1bee9640084322d0f19f1764e4124/raw/57c3d1ee8f1c47acbdcbab47cffb826e7d5eea9a/120x60-chatgpt",
    "https://gist.githubusercontent.com/athersoft/00335c96a7ff7e52a013910b5d657091/raw/89959b566cb1f26909ba3d7d38777d4a4188ee2d/140x70-chatgpt.dat",
    "https://gist.github.com/athersoft/f45eca17f1a5f696d11725d08ed4fdaf/raw/89ab7952b5fc947e3ed6aa4326420658b7dddd33/200x100-chatgpt"
]
instancesGrok = [
    "https://gist.githubusercontent.com/athersoft/3e4fdb3ee806d5cca5c2c1952e1de007/raw/acc503904e7972867591104ff55240c6ddb2dcdb/80x40-grok.dat",
    "https://gist.githubusercontent.com/athersoft/39e316457aa8b8eb03b51ebae423f316/raw/97ea051b3c129ff0c6aeffad92ef8da34cd5693f/100x50-grok.dat",
    "https://gist.githubusercontent.com/athersoft/6daf6a7b4ac2062662601f83e1d2d2bd/raw/e0c5824a7b6cc59734142e8ca66542311b0b1c01/120x60-grok.dat",
    "https://gist.githubusercontent.com/athersoft/9e83d347c7c6516779da64f686573c18/raw/7466c7a863bdfbb6e8ec66a3e730d45b9a84774a/140x70-grok.dat"
]

instancesGemini = [
    "https://gist.github.com/athersoft/e0bfbcdc2bf4beda0ba81daeb87b8a2d/raw/ec7e78f81a177649f83e1fead4054adab51357a0/80x40-gemini.dat",
    "https://gist.github.com/athersoft/b3ce8c66ce3c51e174d81a7ca9eaefd9/raw/916f23718c276601df5a0631c1c59421470767b9/100x50-gemini.dat",
    "https://gist.github.com/athersoft/3853f927779746cb3b8fb8650b8ff4d3/raw/c17057b3fbef7772a522a4891f1fc4be22ff2884/120x60-gemini.dat",
    "https://gist.github.com/athersoft/1b2d3540308e38df4cd8cbaf28348593/raw/db7d39e493d701d12cfe42565152dc01057308bd/140x70-gemini.dat",
    "https://gist.github.com/athersoft/1a26d2dfe533bf2b31fcda682d1b82e7/raw/435db55d97f212e22d8a82bf1d4de3afec6fcd14/100x200-gemini.dat"
]

instancesDeepseek = [
    "https://gist.github.com/athersoft/da76049ae985f515cf3b9759083d6f6d/raw/b85800b5ff3c5a9754cd6af95113e49d2b6c98b9/80x40-deepseek.dat",
    "https://gist.github.com/athersoft/383e7ddf48dcf0d51af9ab5bec757eae/raw/a59a7a3fd7a7cc5b1b610a6b8a9cc5384928e791/100x50-deepseek.dat",
    "https://gist.github.com/athersoft/5544cef94c9382246010c575ff64e8d1/raw/dabe96f65671e4ac6d278d8fe6fe1f51c822b10d/120x60-deepseek.dat",
    "https://gist.github.com/athersoft/63415c7f2205b1c61129ebc1ee3cfcd8/raw/e1635522259d8076bad99dcc0865d11de6b01c96/140x70-deepseek.dat"
]

#Pos 0: 80x40
#Pos 1: 100x50
#Pos 2: 120x60
#Pos 3: 140x70
#pos 4: 200x100 (Solo ChatGPT y Gemini)

# Configuración algoritmo genético
POP_SIZE = 30   #Tamaño de población    
GENERATIONS = 100    #Número de generaciones
N_CORES = 15       #Número de nucleos
MUTATION = 0.15
LICENSE_UUID = "f6447db7-92ab-4268-a78b-70bbb1406031" #Mi licencia de AMPL, no tocar

if __name__ == "__main__":

    DATA_URL = instancesChatGPT[0] #Acá se selecciona la instancia que se evaluará

    print("--- INICIANDO ALGORITMO GENÉTICO HÍBRIDO ---")
    
    # 1. Obtener Datos
    currentInstance = download_data(DATA_URL)
    if not currentInstance:
        sys.exit(1)
        
    num_cds = parse_num_cds(currentInstance)
    print(f"Instancia cargada con {num_cds} CDs candidatos.")

    # 2. Ejecutar GA
    startTime = time.time()
    
    bestChrom, bestCost, costHistory, timeList = geneticAlgorithm(
        modelCode=AMPL_MODEL_CODE,
        dataStr=currentInstance,
        numCds=num_cds,
        popSize=POP_SIZE,
        generations=GENERATIONS,
        nJobs=N_CORES,
        licenseUuid=LICENSE_UUID,
        mutationRate = MUTATION
    )
    
    endTime = time.time()
    
    # 3. Resultados
    print(f"\n--- RESULTADOS FINALES ---")
    print(f"Tiempo Total: {endTime - startTime:.2f} s")
    print(f"Mejor Costo: {bestCost:,.2f}")
    
    cdsAbiertos = [i for i, x in enumerate(bestChrom) if x == 1]
    print(f"CDs Abiertos ({len(cdsAbiertos)}): {cdsAbiertos}")

    plot_convergence(costHistory)