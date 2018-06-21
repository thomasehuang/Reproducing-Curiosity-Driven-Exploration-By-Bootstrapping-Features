README
======

The repository for reproducing the code of the paper *Curiosity-driven Exploration by Bootstrapping Features*.

Our report can be found [here](report.pdf).

RUN INSTRUCTIONS
================
From the `code` directory, enter the `scripts` folder:

```
cd scripts
```

And run `setup.sh`. This will install all of the necessary
```
./setup.sh
```

Then, go back to the `code` directory.
```
cd ..
```

To run our implementation of *Curiosity by Bootstrapping Features* (CBF), run `python3 cbf.py`. Note, CBF has several arguments that can be parsed from the command line. Use `python3 cbf.py -h` for a list of available command line arguments. 

An example of how you might run `cbf.py` to reproduce our results is as follows: 
```
mpirun -np 16 python3 cbf.py --seed 42 --env PongNoFrameskip-v4 --num-timesteps 3000000 --joint-training True &
```
Where `mpirun` allows multiple (in this case, 16) environments to be used in training, and `--joint-training True` updates the embedding network.

The results from each run will be saved in the results/TIMESTAMP directory where TIMESTAMP is the time the run was started.
