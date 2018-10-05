context("RLearner lightgbm")

test_that("fixcubist works", {
  #response + binary classif
  lrn = makeLearner("regr.fixcubist")
  res = holdout(lrn, bh.task)
  expect_equal(c(169, 5), dim(res$pred$data))
  expect_true(sum(res$pred$data$response) != 0)
})

test_that("regr.fixcubist fixes sample problem", {
  #response + binary classif
  lrn = makeLearner("regr.fixcubist")
  td = getTaskData(bh.task)
  colnames(td) = c(paste0("V", sample(c("sample", "Sample", "_sample", ".SAMPLE", "sampled"), ncol(td) - 1, replace = TRUE), "_", seq_len(ncol(td) -1)), "medv")
  res = holdout(lrn, makeRegrTask(data = td, target = "medv"))
  expect_equal(c(169, 5), dim(res$pred$data))
  expect_true(sum(res$pred$data$response) != 0)
})
