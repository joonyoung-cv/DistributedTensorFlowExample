# Distributed TensorFlow Example

## Dependencies
* Ubuntu 16.04
* tmux
* TensorFlow >=1.0.0
* Basic knowledge on tmux is required, e.g., shortcut keys.

## How to run
* For users with single GPU, do the following command:
```
$ bash run_single_gpu.sh
```
In `run_single_gpu.sh`, you can increase the number of workers by modifying `num_workers`.

* For users with multiple GPUs, do the following command:
```
$ bash run_multi_gpu.sh
```
In `run_multi_gpu.sh`, you can increase the number of workers by modifying `num_workers`.
Note that the size of `GPU_ID` in `run_multi_gpu.sh` should be the same as `num_workers`.
For example, if `num_workers` is equal to 2, `GPU_ID` might be `(0 1)`, `(2 4)`, ...

## References
* The network architecture from [@ischlag](https://github.com/ischlag/distributed-tensorflow-example). However, not exactly the same. For example, TensorFlow graph is a little bit different.
* Some functions and ideas come from OpenAI's [universe-starter-agent](https://github.com/openai/universe-starter-agent). However, the code does not suppor GPU. 
