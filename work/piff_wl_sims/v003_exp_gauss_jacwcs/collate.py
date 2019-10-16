import sys
import pickle


outname = sys.argv[1]
data = []
for fname in sys.argv[2:]:
    with open(fname, 'rb') as fp:
        data.append(pickle.load(fp))

with open(outname, 'wb') as fp:
    pickle.dump(data, fp)
