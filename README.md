<!-- PROJECT LOGO -->
<br />
<p align="center">
<a href="#"><img src="assets/Banner.svg" alt="Logo"/></a>
  <h3 align="center">A Practical Debugging Tool for Training Deep Neural Networks</h3>

  <p align="center">
    Experiments
  </p>
</p>

<p align="center">
  <a href="https://github.com/f-dangel/cockpit">Cockpit Package</a> •
  <a href="https://cockpit.readthedocs.io/">Docs</a> •
  <a href="LICENSE.txt">License</a>
</p>

## Experiments

To showcase the capabilities of [Cockpit](https://github.com/f-dangel/cockpit) we performed several experiments illustrating the usefulness of our debugging tool in [our paper](https://arxiv.org/abs/2102.06604). The code for all experiments, as well as the generated data is presented in this repository. For a discussion of those experiments please refer to our paper.

We provide an overivew of all experiments here, splitting them up in experiments that are presented in the main body of our paper and ones that are shown in the appendix.

All necessary requirements (besides [Cockpit](https://github.com/f-dangel/cockpit)) can be found in [requirements.txt](requirements.txt)

If you decide to re-run the code, you may want to delete the `~/temp/data_deepobs` directory, that will be created to share the data sets among experiments, afterwards.

All experiments include some `run.py` file to reproduce the results. Our original results are also stored in each experiment folder in a `results.zip` file. The `plot.py` script produces the Figures shown in the paper.

| Experiment                 | Description                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 01_benchmark               | Benchmark the run time of individual instruments and configurations of Cockpit.<br>Reproduces Figure 6 and 11-15. |
| 02_LINE                    | Illustrative Example of why just monitoring the loss is not enough.<br>Reproduces Figure 1.                       |
| 04_benchmark_memory        | Benchmark the memory consumption of the histograms.<br>Reproduces Figure 10                                       |
| 06_preprocessing           | Study how incorrectly scaled data is represented in Cockpit.<br>Reproduces Figure 3.                              |
| 07_learning_rate_selection | Grid search the learning rate and compare the alpha values vs. final performance.<br>Reproduces Figure 5.         |
| 09_layerwise               | Analyzing the layerwise histogram for two different architectures.<br>Reproduces Figure 4.                        |
| 10_showcase                | Showcase of the full Cockpit for DeepOBS problems.<br>Reproduces Figure 2 and 17.                                 |
| 11_histogram2d             | Benchmark the performance of different implementations for computing the 2D Histogram.<br>Reproduces Figure 16.   |
| 12_alpha_explanation       | Illustrative plot explaining the Alpha Quantity.<br>Reproduces Figure 8.                                          |

## Main Text Experiments

#### Figure 1: Loss is not Enough - [`02_LINE`](experiments/02_LINE/README.md)

![Loss is not Enough](experiments/02_LINE/output/LINE.png)

#### Figure 2: Showcase - [`10_showcase`](experiments/10_showcase/README.md)

![Showcase CIFAR-100](experiments/10_showcase/output/cifar100_allcnnc_log.png)

#### Figure 3: Misscaled Data - [`06_preprocessing`](experiments/06_preprocessing/README.md)

![Misscaled Data](experiments/06_preprocessing/output/exp06.png)

#### Figure 4: Layerwise Histogram - [`09_layerwise`](experiments/09_layerwise/README.md)

![Layerwise Histogram]experiments/09_layerwise/output/exp09.png)

#### Figure 5: Tuning the Learning Rate with Alpha - [`07_learning_rate_selection`](experiments/07_learning_rate_selection/README.md)

![Median Alpha vs Performance](experiments/07_learning_rate_selection/output/median_alpha_vs_performance.png)

#### Figure 6: Run Time Benchmark - [`01_benchmark`](experiments/01_benchmark/README.md)

![Run Time Benchmarks](experiments/01_benchmark/output/exp01.png)

## Appendix Experiments

Figure 7 is not included in this list, as it is a code example. The code is an adapted
version of the [basic](https://github.com/f-dangel/cockpit/blob/main/examples/01_basic_fmnist.py) and [advanced](https://github.com/f-dangel/cockpit/blob/main/examples/02_advanced_fmnist.py) examples from the examples directory and the [documentation](https://cockpit.readthedocs.io/en/latest/examples/01_basic_fmnist.html).

Figure 9 is a conceptual sketch of the gradient tests without any empirical results.

#### Figure 8: Motivational Sketch for Alpha - [`12_alpha_explanation`](experiments/12_alpha_explanation/README.md)

![Alpha Explanation](experiments/12_alpha_explanation/output/alpha_explanation.png)

#### Figure 10: Memory Consumption Histogram - [`04_benchmark_memory`](experiments/04_benchmark_memory/README.md)

![Memory Benchmarks](experiments/04_benchmark_memory/output/exp04.png)

#### Figure 11 - 15: Additional Run Time Benchmarks - [`01_benchmark`](experiments/01_benchmark/README.md)

#### Figure 16: Implementation Performance 2D Histogram - [`11_histogram2d`](experiments/11_histogram2d/README.md)

![Performance Histogram](experiments/11_histogram2d/output/exp11.png)

#### Figure 17: Additional Showcases - [`10_showcase`](experiments/10_showcase/README.md)

![Showcase MNIST LogReg](experiments/10_showcase/output/mnist_logreg_log.png)
![Showcase Quadratic Deep](experiments/10_showcase/output/quadratic_deep_log.png)
