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

### Test recall for 1M dataset

```
make build
make run-1m
make run-recall-1m
```

## Submit

install reprozip before submit
```
pip install reprozip
```
pack code
```
make clean && rm -rf .reprozip-trace && rm -f submission.rpz
reprozip trace bash ./run.sh
reprozip pack submission.rpz
```
submit the `submission.rpz`.

Note: Submission will take some time to wait.