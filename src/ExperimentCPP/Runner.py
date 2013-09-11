import subprocess

if __name__ == "__main__":
    algoNames = ["rgabor", "hog", "rhog", "gist", "cm", "ocm", "hitmap", "sc", "hoosc"]

    for algoName in algoNames:
        p = subprocess.Popen("ExperimentCPP.exe " + algoName)
        p.communicate()