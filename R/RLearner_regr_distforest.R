#' @export
makeRLearner.regr.distforest = function() {
  makeRLearnerRegr(
    cl = "regr.distforest",
    package = c("ranger", "gamlss", "parallelMap"),
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "num.trees", lower = 1L, default = 500L),
      makeIntegerLearnerParam(id = "mtry", lower = 1L),
      makeNumericLearnerParam(id = "mtry.perc", lower = 0, upper = 1),
      makeIntegerLearnerParam(id = "min.node.size", lower = 1L, default = 5L),
      makeLogicalLearnerParam(id = "replace", default = TRUE),
      makeNumericLearnerParam(id = "sample.fraction", lower = 0L, upper = 1L),
      makeNumericVectorLearnerParam(id = "split.select.weights", lower = 0, upper = 1),
      makeUntypedLearnerParam(id = "always.split.variables"),
      makeDiscreteLearnerParam("respect.unordered.factors", values = c("ignore", "order", "partition"), default = "ignore"),
      makeDiscreteLearnerParam(id = "importance", values = c("none", "impurity", "permutation"), default = "none", tunable = FALSE),
      makeLogicalLearnerParam(id = "write.forest", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "scale.permutation.importance", default = FALSE, requires = quote(importance == "permutation"), tunable = FALSE),
      makeIntegerLearnerParam(id = "num.threads", lower = 1L, when = "both", tunable = FALSE),
      makeLogicalLearnerParam(id = "save.memory", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "verbose", default = TRUE, when = "both", tunable = FALSE),
      makeIntegerLearnerParam(id = "seed", when = "both", tunable = FALSE),
      makeDiscreteLearnerParam(id = "splitrule", values = c("variance", "extratrees", "maxstat"), default = "variance"),
      makeIntegerLearnerParam(id = "num.random.splits", lower = 1L, default = 1L, requires = quote(splitrule == "extratrees")),
      makeNumericLearnerParam(id = "alpha", lower = 0L, upper = 1L, default = 0.5, requires = quote(splitrule == "maxstat")),
      makeNumericLearnerParam(id = "minprop", lower = 0L, upper = 1L, default = 0.1, requires = quote(splitrule == "maxstat")),
      makeLogicalLearnerParam(id = "keep.inbag", default = FALSE, tunable = FALSE)
    ),
    par.vals = list(num.threads = 1L, verbose = FALSE, respect.unordered.factors = "order"),
    # FIXME jitter not found as parameter for km() or km.predict(). par.vals and LearnerParam are the same here.
    properties = c("numerics", "factors", "ordered", "oobpreds", "featimp", "se"),
    name = "DistributionForest",
    short.name = "distforest",
    note = "",
    callees = ""
  )
}

#' @export
trainLearner.regr.distforest = function(.learner, .task, .subset, .weights, keep.inbag = NULL, mtry, mtry.perc, ...) {
 tn = getTaskTargetNames(.task)
  if (missing(mtry)) {
    if (missing(mtry.perc)) {
      mtry = floor(sqrt(getTaskNFeats(.task)))
    } else {
      mtry = max(1, floor(mtry.perc * getTaskNFeats(.task)))
    }
  }
  keep.inbag = if (is.null(keep.inbag)) FALSE else keep.inbag
  keep.inbag = if (.learner$predict.type == "se") TRUE else keep.inbag
  res = list()
  d = getTaskData(.task, .subset)
  res$model = ranger::ranger(formula = NULL, dependent.variable = tn, data = d, keep.inbag = keep.inbag, mtry = mtry, ...)
  res$train.preds = predict(res$model,d, type = "terminalNodes")$predictions
  res$train.data = d
  res$train.target = getTaskTargetNames(.task)
  return(res)
}

#' @export
predictLearner.regr.distforest = function(.learner, .model, .newdata, jitter, ...) {
  train.preds = .model$learner.model$train.preds
  train.data = .model$learner.model$train.data
  train.target.name = .model$learner.model$train.target
  # forest.preds = predict(object = .model$learner.model$model, data = .newdata, type = "terminalNodes", ...)$predictions
  forest.preds = predict(object = .model$learner.model$model, data = .newdata, type = "terminalNodes")$predictions
  if (!is.matrix(forest.preds))
    forest.preds = matrix(forest.preds, nrow = 1)
  ctrl = gamlss::gamlss.control(trace = FALSE)
  gamlss.preds = parallelMap::parallelMap(function(i, train.target.name, train.preds, forest.preds, train.data, ctrl, newdata) {
    f = reformulate("1", train.target.name)
    w = apply(train.preds, MARGIN = 1, function(x) sum(x == forest.preds[i,]))
    tmp = gamlss::gamlss(f, data = train.data, weights = w, control = ctrl)
    unlist(gamlss::predictAll(tmp, newdata = newdata[i,,drop=FALSE], data = train.data, type = "response"))
  }, i = seq_len(nrow(.newdata)), 
    more.args = list(train.target.name = train.target.name, train.preds = train.preds, forest.preds = forest.preds, train.data = train.data, ctrl = ctrl, newdata = .newdata))
  gamlss.preds = do.call(rbind, gamlss.preds)

  if (.learner$predict.type == "se") {
    return(gamlss.preds)
  } else {
    return(gamlss.preds[,1])
  }
}

registerS3method("makeRLearner", "regr.distforest", makeRLearner.regr.distforest)
registerS3method("trainLearner", "regr.distforest", trainLearner.regr.distforest)
registerS3method("predictLearner", "regr.distforest", predictLearner.regr.distforest)
