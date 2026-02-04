import time
import os
import sys
from amplpy import AMPL, ampl_notebook

# --- CLASE PARA SILENCIAR SALIDA ---
class Silence:
    """Context Manager para silenciar stdout/stderr a nivel de OS."""
    def __enter__(self):
        self.fd_out = os.dup(1)
        self.fd_err = os.dup(2)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.fd_out, 1)
        os.dup2(self.fd_err, 2)
        os.close(self.devnull)
        os.close(self.fd_out)
        os.close(self.fd_err)

# Variable global para el worker
workerAmpl = None

def initWorker(modelFile, dataFile, licenseUuid, gurobiOptions):
    """Inicializa la instancia de AMPL en un proceso paralelo."""
    global workerAmpl
    
    # Usamos Silence durante la carga para evitar logs iniciales
    with Silence():
        try:
            try:
                workerAmpl = ampl_notebook(modules=["gurobi"], license_uuid=licenseUuid)
            except:
                workerAmpl = AMPL()
        
            # Lectura de archivos físicos
            workerAmpl.read(modelFile)
            workerAmpl.readData(dataFile)
            
            # Opciones de silencio AMPL
            workerAmpl.setOption("solver_msg", 0)
            workerAmpl.setOption("show_stats", 0)
            
            # Configuración Solver
            workerAmpl.setOption("solver", "gurobi")
            
            # Opciones de silencio Gurobi
            cleanOpts = f"{gurobiOptions} outlev=0 timing=0 logfile=''"
            workerAmpl.option["gurobi_options"] = cleanOpts
            
        except Exception:
            workerAmpl = None

def solveWorker(chromosome):
    """Evalúa un cromosoma."""
    global workerAmpl
    
    if workerAmpl is None:
        return float('inf'), 0.0
        
    if sum(chromosome) == 0:
        return float('inf'), 0.0
        
    try:
        fixCmds = [f"fix Z[{i}] := {x};" for i, x in enumerate(chromosome)]
        workerAmpl.eval("".join(fixCmds))
        
        t0 = time.time()
        
        # Silenciamos la ejecución del solve
        with Silence():
            workerAmpl.solve()
            
        t1 = time.time()
        
        solveResult = workerAmpl.getValue("solve_result")
        
        if solveResult == "solved" or solveResult == "limit":
            objValue = workerAmpl.getObjective("TotalCost").value()
        else:
            objValue = float('inf')
            
        return objValue, (t1 - t0)
        
    except Exception:
        return float('inf'), 0.0