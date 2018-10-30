context("classif_multinomial_logistic")

test_that("classif_multinomial_logistic", {
  requirePackagesOrSkip("mxnet", default.method = "load")

  lrn = makeLearner("classif.mx.multinomiallogistic", num.round = 50)
  res = resample(lrn, iris.task, cv3, acc)
  expect_class(res, "ResampleResult")
  expect_true(res$aggr > 0.5)
})
