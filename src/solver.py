import time
from amplpy import AMPL, ampl_notebook

# Variable global para cada proceso
workerAmpl = None

def initWorker(modelFile, dataFile, licenseUuid, gurobiOptions):
    """Inicializa AMPL leyendo archivos físicos."""
    global workerAmpl
    try:
        # Intenta cargar licencia académica
        workerAmpl = ampl_notebook(modules=["gurobi"], license_uuid=licenseUuid)
    except:
        # Fallback local (usará HiGHS si no encuentra Gurobi o falla licencia)
        workerAmpl = AMPL()
    
    try:
        workerAmpl.read(modelFile)
        workerAmpl.readData(dataFile)
        
        # 1. Silenciar mensaje de éxito ("Optimal solution...")
        workerAmpl.setOption("solver_msg", 0)
        
        # Configuración del solver
        workerAmpl.setOption("solver", "gurobi")
        
        # 2. Asegurar que Gurobi no imprima log de iteraciones
        # Añadimos outlev=0 a las opciones que vienen de main
        full_gurobi_opts = f"{gurobiOptions} outlev=0"
        workerAmpl.option["gurobi_options"] = full_gurobi_opts
        
        workerAmpl.setOption("os_options", "outlev=0")
        
    except Exception as e:
        print(f"FATAL ERROR en initWorker: {e}")
        workerAmpl = None

def solveWorker(chromosome):
    """Resuelve un individuo."""
    global workerAmpl
    
    if workerAmpl is None:
        return float('inf'), 0.0
        
    if sum(chromosome) == 0:
        return float('inf'), 0.0
        
    try:
        # Fijar variables Z
        fixCommands = [f"fix Z[{i}] := {x};" for i, x in enumerate(chromosome)]
        workerAmpl.eval("".join(fixCommands))
        
        t0 = time.time()
        workerAmpl.solve()
        t1 = time.time()
        
        solveResult = workerAmpl.getValue("solve_result")
        
        if solveResult == "solved" or solveResult == "limit":
            objValue = workerAmpl.getObjective("TotalCost").value()
        else:
            objValue = float('inf')
            
        return objValue, (t1 - t0)
        
    except Exception as e:
        return float('inf'), 0.0