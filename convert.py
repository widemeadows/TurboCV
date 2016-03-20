import sys


def main(argv=sys.argv):
    if len(sys.argv) <= 1:
        print "Missing Arguments..."
        exit()

    input = open(sys.argv[1])

    tokens = input.readline().strip().split(' ')
    nVec = int(tokens[0])
    nDim = int(tokens[1])

    tokens = input.readline().strip().split(' ')
    if len(tokens) == nDim:  # words of local features
        for i in range(nVec - 1):
            input.readline()

        tokens = input.readline().strip().split(' ')
        nVec = int(tokens[0])
        nDim = int(tokens[1])

        tokens = input.readline().strip().split(' ')

    if len(sys.argv) >= 3 and sys.argv[2] == '-svm':
        output = open(sys.argv[1] + '.svm', 'w')

        def write_line(tokens):
            output.write(tokens[0])
            for j in range(nDim):
                output.write(' ' + str(j + 1) + ':' + tokens[j + 1])
            output.write('\n')

        write_line(tokens)
        for i in range(nVec - 1):
            write_line(input.readline().strip().split(' '))

        output.close()
    else:
        import numpy as np
        import scipy.io as sio

        x = np.zeros(shape=(nDim, nVec), dtype=np.float64)
        y = np.zeros(shape=(nVec,), dtype=np.int32)

        x[:, 0] = [float(token) for token in tokens[1:]]
        y[0] = int(tokens[0])

        for i in range(nVec - 1):
            tokens = input.readline().strip().split(' ')
            x[:, i + 1] = [float(token) for token in tokens[1:]]
            y[i + 1] = int(tokens[0])

        sio.savemat(sys.argv[1] + '.mat', {'x': x, 'y': y})

    input.close()


if __name__ == "__main__":
    main()