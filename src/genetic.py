import random
import multiprocessing
import os
from .solver import initWorker, solveWorker

def createBinaryChromosome(numCds):
    chrom = [random.choice([0, 1]) for _ in range(numCds)]
    if sum(chrom) == 0:
        chrom[random.randint(0, numCds - 1)] = 1
    return chrom

def tournamentSelection(population, fitnesses, k=2):
    selectedIndices = random.sample(range(len(population)), k)
    selectedFitnesses = [fitnesses[i] for i in selectedIndices]
    bestLocalIndex = selectedIndices[selectedFitnesses.index(min(selectedFitnesses))]
    return population[bestLocalIndex]

def crossoverBinary(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutateBinary(chromosome, mutationRate=0.05):
    newChrom = chromosome[:]
    for i in range(len(newChrom)):
        if random.random() < mutationRate:
            newChrom[i] = 1 - newChrom[i] 
    if sum(newChrom) == 0:
         newChrom[random.randint(0, len(newChrom) - 1)] = 1
    return newChrom

def geneticAlgorithmParallel(modelCode, dataStr, numCds, popSize=20, generations=10, mutationRate=0.1, elitism=True, nJobs=2, licenseUuid=""):
    
    # Escribir archivos temporales para los workers
    tempModelName = "temp_model.mod"
    tempDataName = "temp_data.dat"
    
    with open(tempModelName, "w") as f:
        f.write(modelCode)
    with open(tempDataName, "w") as f:
        f.write(dataStr)
        
    print(f"Archivos temporales creados: {tempModelName}, {tempDataName}")

    fitnessTimeList = []
    fitnessCache = {}
    gurobiOpts = "NonConvex=2 MIPGap=0.05"

    print(f"Iniciando Pool con {nJobs} procesos...")
    
    pool = multiprocessing.Pool(
        processes=nJobs,
        initializer=initWorker,
        initargs=(tempModelName, tempDataName, licenseUuid, gurobiOpts)
    )
    
    try:
        def evaluatePopulation(currentPop):
            costs = [0.0] * len(currentPop)
            times = [0.0] * len(currentPop)
            toCalculate = [] 
            
            for idx, ind in enumerate(currentPop):
                chromKey = tuple(ind)
                if chromKey in fitnessCache:
                    costs[idx] = fitnessCache[chromKey]
                    times[idx] = 0.0 
                else:
                    toCalculate.append((idx, ind))
            
            if toCalculate:
                chromosomesOnly = [item[1] for item in toCalculate]
                results = pool.map(solveWorker, chromosomesOnly)
                
                for i, (val, t) in enumerate(results):
                    originalIdx = toCalculate[i][0]
                    chromKey = tuple(chromosomesOnly[i])
                    fitnessCache[chromKey] = val
                    costs[originalIdx] = val
                    times[originalIdx] = t
                    
            return costs, times

        # --- GA LOOP ---
        print("Generando población inicial...")
        population = [createBinaryChromosome(numCds) for _ in range(popSize)]
        
        fitnesses, times = evaluatePopulation(population)
        fitnessTimeList.extend(times)
        
        bestIdx = fitnesses.index(min(fitnesses))
        bestSolution = population[bestIdx][:]
        bestCost = fitnesses[bestIdx]
        bestCostHistory = [bestCost]
        
        print(f"Gen 0 | Mejor costo: {bestCost:,.2f} | Cache: {len(fitnessCache)}")
        
        for gen in range(generations):
            newPopulation = []
            if elitism: newPopulation.append(bestSolution[:])
            
            while len(newPopulation) < popSize:
                p1 = tournamentSelection(population, fitnesses)
                p2 = tournamentSelection(population, fitnesses)
                c1, c2 = crossoverBinary(p1, p2)
                c1 = mutateBinary(c1, mutationRate)
                c2 = mutateBinary(c2, mutationRate)
                newPopulation.extend([c1, c2])
            
            population = newPopulation[:popSize]
            
            fitnesses, times = evaluatePopulation(population)
            fitnessTimeList.extend(times)
            
            minFit = min(fitnesses)
            if minFit < bestCost:
                bestCost = minFit
                bestSolution = population[fitnesses.index(minFit)][:]
                print(f"Gen {gen+1} | ¡Nuevo Récord!: {bestCost:,.2f}")
                
            bestCostHistory.append(bestCost)
            
            if (gen+1) % 5 == 0:
                print(f"Gen {gen+1}/{generations} completada.")

    except KeyboardInterrupt:
        print("\nInterrupción detectada. Cerrando pool...")
        
    finally:
        pool.close()
        pool.join()
        # Limpieza de archivos temporales (opcional)
        # if os.path.exists(tempModelName): os.remove(tempModelName)
        # if os.path.exists(tempDataName): os.remove(tempDataName)
        
    return bestSolution, bestCost, bestCostHistory, fitnessTimeList