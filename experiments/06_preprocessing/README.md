# Instructions to reproduce this experiment

![Debugging data pre-processing](output/exp06.png)

## Sample images

1. Run experiment: `python run_samples_cifar10.py` and `python run_samples_imagenet.py`. Find the images in `output/fig_samples`.
2. Clean up or start over: `bash clean_samples.sh`

## Gradient element distribution

1. (Optional) Extract `results.zip` to use the original data.
2. Run experiment: `python run_histograms_cifar10.py` and `python run_histograms_imagenet.py`.
3. Plot the results: `python plot_histograms.py`. Find the images in `output/fig_histogram`.
4. Clean up or start over: `bash clean_histograms.sh`
