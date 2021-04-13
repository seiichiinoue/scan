# Implementation of A Bayesian Model of Diachronic Meaning Change

## Environment

using docker to run code in this repository.

```
$ docker build -t boost .
$ docker run -it -b [LOCAL_PATH]:[CONTAINER_PATH] boost
```

## Run

compile and train.

```
$ make
$ ./scan -burn_in_period=150 -ignore_word_count=3 -data_path=PATH_TO_DATA -save_path=PATH_TO_MODEL
```

## References

- [A Bayesian Model of Diachronic Meaning Change. (2016). L. Frermann and M. Lapata.](https://www.aclweb.org/anthology/Q16-1003.pdf)