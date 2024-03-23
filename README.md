# Abstract

Given a set of parts, create a valid construction manual for building those parts together. The output target is a graph that represents which parts are to be connected.

The data stems from a real-world data set. 
`data/graphs.dat` contains 11.159 graphs representing constructions. Notes about these graphs:

* Undirected, unweighted, non-cyclic and connected without self-loops.
* They can contain multiple instances of the same part
* 1.089 different parts; parts are pseudonymized

<img width="982" alt="image" src="https://github.com/inkenlara/ai-building-constructions/assets/74065011/829aa317-c5f5-4d10-9f8a-dc2f86d3506b">


# Technical Docs

This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).
(The project can be upgraded to a higher Python version if required. )

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.   
`conda create -n ai-project python=3.8`

2. Activate the new environment
`conda activate ai-project`

3. Install the dependencies:  
`conda install -c conda-forge --file requirements.txt` 


## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html). 
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should automatically search all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.
