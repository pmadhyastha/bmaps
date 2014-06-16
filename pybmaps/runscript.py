import glob
import sepnexec
import sys

qdev = sys.argv[1]
cdev = sys.argv[2]
constr = sys.argv[3]
trfile = sys.argv[4]
tqdev = sys.argv[5]
tconstr = sys.argv[6]
folder = [x.strip() for x in open(trfile)]

for direc in glob.glob('fobos*'):
    if direc in folder:
        status = sepnexec.compute(direc, tqdev, cdev, tconstr)
    else:
        status = sepnexec.compute(direc, tqdev, cdev, tconstr)

    if status == 'Done':
        continue
    else:
        sys.exit('Error')


