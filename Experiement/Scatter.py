import sys
import pylab as plot

if __name__ == "__main__":
    pointFileName = "Y.txt"
    labelFileName = "labels.txt"

    if len(sys.argv) == 3:
        pointFileName = sys.argv[1]
        labelFileName = sys.argv[2]
    elif len(sys.argv) != 1:
        print "Unsupported params!"
        exit();

    x = []
    y = []
    pFile = open(pointFileName)
    while True:
        line = pFile.readline()
        if not line:
            break
        
        tokens = line.strip().split(' ')
        x.append(float(tokens[0]))
        y.append(float(tokens[1]))

    pFile.close()

    labels = []
    lFile = open(labelFileName)    
    line = lFile.readline()
    for token in line.strip().split(' '):
        labels.append(float(token))

    lFile.close()

    plot.scatter(x, y, 20, labels)
    plot.show()