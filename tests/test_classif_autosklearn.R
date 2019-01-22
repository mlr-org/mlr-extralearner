context("autosklearn")

test_that("classif_ranger.pow", {
  require("reticulate")
  use_virtualenv("~/Documents/mlr_repos/mlr-extralearner/venv")

  lrn = makeLearner("classif.autosklearn", time_left_for_this_task = 20L, per_run_time_limit = 10L)
  mod = train(lrn, iris.task)
  expect_class(mod, "WrappedModel")

  prd = predict(mod, iris.task)
  expect_class(prd, "PredictionClassif")

  res = resample(lrn, pid.task, cv3)
  expect_class(res, "ResampleResult")


  lrn = makeLearner("classif.autosklearn", time_left_for_this_task = 20L, per_run_time_limit = 10L,
    include_estimators = list("random_forest", "libsvm_svc"))
  mod = train(lrn, iris.task)
  expect_class(mod, "WrappedModel")
})
