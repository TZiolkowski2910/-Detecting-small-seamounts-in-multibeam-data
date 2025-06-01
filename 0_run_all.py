import subprocess
import os

# Liste der Skripte, die ausgeführt werden sollen
scripts = [
    "1_segment_images.py",
    "2_extractFeatureVectors_VGG16.py",
    "3_extractFeatureVectors_ResNet50.py",
    "4_extractFeatureVectors_InceptionV3.py",
    "5_extractFeatureVectors_MobileNetV2.py",
    "6_kmeans_clustering.py",
    "7_agglo_clustering.py"
]

# Verzeichnis, in dem die Skripte liegen
script_dir = r"C:\Users\Tobias\Arbeitsaufgaben\Paper schreiben\Seamounts\python"

# Funktion zum Ausführen eines Skripts
def run_script(script_name):
    script_path = os.path.join(script_dir, script_name)
    try:
        result = subprocess.run(["python3", script_path], check=True, text=True, capture_output=True)
        print(f"Erfolgreich ausgeführt: {script_name}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei der Ausführung von: {script_name}")
        print(e.stderr)

# Alle Skripte der Reihe nach ausführen
for script in scripts:
    run_script(script)

print("Alle Skripte wurden erfolgreich ausgeführt.")
