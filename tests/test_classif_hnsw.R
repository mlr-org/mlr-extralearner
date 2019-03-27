context("classif_hnsw")

test_that("classif_hnsw", {
  # requirePackagesOrSkip("RcppHNSW", default.method = "load")

  parset.list = list(
    list(),
    list(k = 7, ef_construction = 5),
    list(k = 7, ef_construction = 5, ef = 5),
    list(k = 7, ef_construction = 5, ef = 200),
    list(k = 9, ef_construction = 70, M = 2),
    list(k = 11, ef_construction = 5, M = 100),
    list(k = 11, ef_construction = 5, M = 1000),
    list(k = 1, ef_construction = 5),
    list(k = 50, ef_construction = 1000, ef = 1000),
    list(k = 4, distance = "cosine"),
    list(k = 4, distance = "l2"),
    list(k = 4, distance = "ip"),
    list(k = 1)
  )

  for (i in seq_along(parset.list)) {
    lrn = makeLearner("classif.RcppHNSW")
    lrn = setHyperPars(lrn, par.vals = parset.list[[i]])
    if (i %% 2 == 0) lrn = setPredictType(lrn , "prob")
    r2 = resample(lrn, iris.task, hout)
    r1 = resample(lrn, pid.task, hout)
    expect_class(classes = "ResampleResult", r1)
    expect_class(classes = "ResampleResult", r2)
  }

})
