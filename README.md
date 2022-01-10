# nonlinear

This uses a nonlinear solver based on the method by Yamashita et. al. to search for Mutually Unbiased Bases (MUBs).

### Dependencies

This code requires OpenMP for parallelisation and Eigen for matrix calculations, which can both be installed using the following command on Debian-based distributions:
```bash
sudo apt-get install libomp-dev libeigen3-dev
```

You also need make and g++ (in case they aren't installed already):
```bash
sudo apt-get install make g++
```

### Compilation

To compile the code, first clone this repository somewhere:
```bash
git clone https://github.com/Lumorti/nonlinear
```

Then enter this directory and compile:
```bash
cd nonlinear
make
```

### Usage

To run the code for the case of dimension-2 with 3 bases (d2n2):
```bash
./nonlin -d 2 -n 3
```

See the help for other command-line arguments:
```bash
./nonlin -h
```
