# nonlinear

This uses a nonlinear solver based on the method by Yamashita et. al. to search for Mutually Unbiased Bases (MUBs).

### Installation

This code requires OpenMP for parallelisation and Eigen for matrix calculations, which can both be installed using the following command on debian-based distributions:

```bash
sudo apt-get install make libomp-dev libeigen3-dev
```

To compile the code, first clone this repository somewhere:

```bash
git clone https://github.com/Lumorti/nonlinear
```

Then enter this directory and compile:

```bash
make
```

### Usage

To run the code for the simple case of dimension-2 with 2 bases (d2n2):

```bash
./nonlin
```

This time for d3n4:

```bash
./nonlin -d 3 -n 4
```

See the help for other command-line arguments:

```bash
./nonlin -h
```
