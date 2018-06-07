# mlr-extralearner

This repository contains additional learner for [mlr](https://github.com/mlr-org/mlr) that we don't want to maintain in the mlr directly.


**Warning**: These learners can be untested, unstable or unfinished. Use at your own risk.

To easily use them, just source the path to the `raw` github link, e.g.,

```{r}
library(mlr)
source("https://raw.githubusercontent.com/mlr-org/mlr-extralearner/master/R/RLearner_classif_lightgbm.R")
```


If you have learner feel free to add them here by creating a [pull request](https://github.com/mlr-org/mlr-extralearner/compare). 
