# mnist_pysh_ca & pysh_ca
<!-- <img src="https://github.com/tm224/PushGP-CA-Image-Class/blob/main/demo.gif" width="200" /> -->

pysh_ca is a package combining Push genetic programming and a cellular automata grid in order to make a classification system.

This project specifically uses the MNIST data set to test classification performance using a custom implementation of pysh_ca called mnist_pysh_ca.

## How does it work?

The system works by creating a population of genetic programs that are evolved over time. Each program creates a CA grid based on one sample of the data set (called the initial condition) and using the program as the update rule runs the CA for a number of steps. After those steps the output of the grid is used as the classification of that sample.

The idea is that each generation will find the best performing individuals and through a method of crossover and mutation the program's fitness improved.

This project contains the new package designed for implementation when making a system, called pysh_ca, and the implementation to be used with the MNIST for testing, which is called mnist_pysh_ca.

## pysh_ca

pysh_ca is designed to allow for any labeled data set as well as custom initial conditions for the CA grid and custom classification functions after a certain number of time steps. This package is designed to be adapted for use outside of this project and is currently in version 0.0.1.

pysh_ca uses the PyshGP package for its Push genetic programming implementation and cellular-automaton package for its cellular automata implementation. Both packages have been extended by the pysh_ca package and will need to be installed using pip before using the package.

The package is broken into two directories, pyshgp and ca, depending on what package the file extends.

The process of implementing a pysh_ca project is similar to a PyshGP project, but instead of creating a PushEstimator you have to create a CAEstimator. An example of the implementation of pysh_ca with the MNIST data set can be seen in the mnist_pysh_ca.py file.

## mnist_pysh_ca

mnist_pysh_ca is a demo implementation of pysh_ca designed to allow a user to play around with it without having to implement anything further.

It works by creating a CA based on a sample from the MNIST data set and the classification of the CA is the average intensity of the grid after its last step.

Currently it features a CLI interface called using:

```bash
python main.py --help
```

It contains training, testing, and simplifcation modes, each one with various arguments.
If you just want to try out the program and see how it works try running:

```bash
python main.py -lf output/best-individual.json -m testing -pic True -d 1,2 -s 10 -c 1
```
This will test the best individual I found and output a txt file showing its output and error as well as an animated gif showing how the Push update rule effects the grid.

There is also some demo files in the output folder.
