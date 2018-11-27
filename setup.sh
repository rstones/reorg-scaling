#!/bin/bash

# pre-requisites: python 3.6, virtualenv

PYTHON36=$1
if [ ! $PYTHON36 ]
then
	PYTHON36="/usr/bin/python3.6"
fi

# checking for python 3.6 
if [ ! -e $PYTHON36 ]
then
	echo "The python 3.6 installation provided or the default /usr/bin/python3.6 does not exist. Exiting..."
	exit
fi

# checking for virtualenv
VENV=$(which virtualenv)
if [ ! $VENV ]
then
	echo "You must install virtualenv before continuing. Exiting..."
	exit
fi

# set up the virtualenv...
virtualenv -p $PYTHON36 venv
source venv/bin/activate
pip install numpy scipy numba
pip install git+https://github.com/rstones/counting_statistics.git
pip install git+https://github.com/rstones/quant_mech.git

# make data directory
mkdir data
