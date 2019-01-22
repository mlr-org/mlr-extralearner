context("classif_ranger.pow")

test_that("classif_ranger.pow", {
  requirePackagesOrSkip("ranger", default.method = "load")

  parset.list = list(
    list(),
    list(mtry = 1),
    list(mtry = 0.1),
    list(mtry = 0.5),
  )

  old.probs.list = list()

  for (i in seq_along(parset.list)) {
    parset = parset.list[[i]]
    parset = c(parset, list(data = binaryclass.train, formula = binaryclass.formula, write.forest = TRUE, probability = TRUE, respect.unordered.factors = TRUE))
    set.seed(getOption("mlr.debug.seed"))
    m = do.call(ranger::ranger, parset)
    p  = predict(m, data = binaryclass.test)
    old.probs.list[[i]] = p$predictions[, 1]
  }

})
