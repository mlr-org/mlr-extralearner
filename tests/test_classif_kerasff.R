context("classif_kerasff")

test_that("classif_kerasff", {
  # requirePackagesOrSkip("RcppHNSW", default.method = "load")
  lrn = makeLearner("classif.kerasff")
  r = resample(lrn, iris.task, hout)
  expect_class(r, "ResampleResult")

  lrn = makeLearner("classif.kerasff", predict.type = "prob")
  r = resample(lrn, iris.task, hout)
  expect_class(r, "ResampleResult")


  lrn = makeLearner("classif.kerasff", optimizer = "sgd", momentum = 0.7, decay = 0.01, lr = 0.01)
  r = resample(lrn, iris.task, hout)
  expect_class(r, "ResampleResult")

  lrn = makeLearner("classif.kerasff", early_stopping_patience = 0)
  r = resample(lrn, iris.task, hout)

  lrn = makeLearner("classif.kerasff", learning_rate_scheduler = TRUE)
  r = resample(lrn, iris.task, hout)
})
