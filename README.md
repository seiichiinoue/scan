# Implementation of A Bayesian Model of Diachronic Meaning Change

## Environment

using docker to run code in this repository.

```
$ docker build -t boost .
$ docker run -it -b [LOCAL_PATH]:[CONTAINER_PATH] boost
```

## Data

`documents.txt`: each line correspond to word-specific snippet

```
d_0_context_{-I} d_0_context_{-I+1} ... d_0_context_{-1} d_0_context_{+1} d_0_context_{+2} ... d_0_context_{+I}
d_1_context_{-I} d_1_context_{-I+1} ... d_1_context_{-1} d_1_context_{+1} d_1_context_{+2} ... d_1_context_{+I}
...
d_N_context_{-I} d_N_context_{-I+1} ... d_N_context_{-1} d_N_context_{+1} d_N_context_{+2} ... d_N_context_{+I}
```

`time_labels.txt`: each line correspond to time label of corresponding snippet.

```
0
1
...
T
```

## Run

compile and train.

```
$ make
$ ./scan -num_iteration=1000 -burn_in_period=150 -ignore_word_count=3 -data_path=PATH_TO_DATA -save_path=PATH_TO_MODEL
```

## References

- [A Bayesian Model of Diachronic Meaning Change. (2016). L. Frermann and M. Lapata.](https://www.aclweb.org/anthology/Q16-1003.pdf)