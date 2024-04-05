# SIGMOD programming contest 2024

## Run
download dataset from website, put it into folder `data` and then `bash run.sh`.

### Run other data
compile
```
make build
```
run 1m data
```
make run-1m
```
or run 10m data
```
make run-10m
```

## Submit

install reprozip before submit
```
pip install reprozip
```
pack code
```
make clean
reprozip trace bash ./run.sh
reprozip pack submission.rpz
```
submit the `submission.rpz`.

Note: Submission will take some time to wait.