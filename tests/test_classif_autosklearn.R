context("autosklearn")

test_that("classif_ranger.pow", {
  require("reticulate")
  use_virtualenv("~/Documents/mlr_repos/mlr-extralearner/venv")

  lrn = makeLearner("classif.autosklearn", time_left_for_this_task = 20L)
  mod = train(lrn, iris.task)
  expect_class(mod, "WrappedModel")

  prd = predict(mod, iris.task)
  expect_class(prd, "PredictionClassif")

  resample(lrn, pid.task, cv3)
})
