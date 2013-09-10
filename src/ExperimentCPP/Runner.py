import subprocess

if __name__ == "__main__":
    algoNames = ["hog", "rhog", "hoosc", "sc", "gist", "cm", "ocm", "hitmap"]

    for algoName in algoNames:
        p = subprocess.Popen("ExperimentCPP.exe " + algoName)
        p.communicate()