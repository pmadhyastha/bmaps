#/bin/bash 

source $HOME/local/virtualenv/bin/activate
python setup.py build_ext --inplace 
export PYTHONPATH=`pwd`:$PYTHONPATH

