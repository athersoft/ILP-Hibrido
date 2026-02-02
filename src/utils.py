import requests
import re
import matplotlib.pyplot as plt

def download_data(url):
    print(f"Descargando archivo .dat desde Gist...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error descargando: {e}")
        return None

def parse_num_cds(ampl_data):
    match = re.search(r'set\s+I\s*:=\s*(.*?);', ampl_data, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        cdList_ids = content.split()
        return len(cdList_ids)
    return 0

def plot_convergence(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o')
    plt.title("Convergencia GA Híbrido")
    plt.xlabel("Generación")
    plt.ylabel("Costo")
    plt.grid(True)
    plt.show()