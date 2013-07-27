import sys


def main(argv=sys.argv):
    input = open(sys.argv[1])
    output = open(sys.argv[1] + '.svm', 'w')
    
    tokens = input.readline().strip().split(' ')
    nVec = int(tokens[0])
    nDim = int(tokens[1])
    for i in range(nVec):
        tokens = input.readline().strip().split(' ')
        
        output.write(tokens[0])
        for j in range(nDim):
            output.write(' ' + str(j + 1) + ':' + tokens[j + 1])
        output.write('\n')

    input.close()
    output.close()


if __name__ == "__main__":
    main()