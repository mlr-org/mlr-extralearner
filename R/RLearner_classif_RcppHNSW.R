#' @export
makeRLearner.classif.RcppHNSW = function() {
  makeRLearnerClassif(
    cl = "classif.RcppHNSW",
    package = "RcppHNSW",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "k", lower = 1L, default = 1L, when = "predict", upper = 50L),
      makeDiscreteLearnerParam(id = "distance", values = c("euclidean", "l2", "cosine", "ip"), default = "euclidean"),
      makeIntegerLearnerParam(id = "M", lower = 2, upper = Inf, default = 16),
      makeIntegerLearnerParam(id = "ef", lower = 1, upper = Inf, default = 10, when = "predict"),
      makeIntegerLearnerParam(id = "ef_construction", lower = 1, upper = Inf, default = 200),
      makeLogicalLearnerParam(id = "verbose", default = FALSE, tunable = FALSE)
    ),
    par.vals = list(k = 1, M = 16, ef_construction = 200),
    properties = c("twoclass", "multiclass", "prob", "numerics"),
    name = "Approximate neares neighbours",
    short.name = "hnsw",
    callees = "hnsw"
  )
}

#' @export
trainLearner.classif.RcppHNSW = function(.learner, .task, .subset, .weights = NULL, ...) {
  data = getTaskData(.task, .subset, target.extra = TRUE)
  pv = list(...)
  ann = RcppHNSW::hnsw_build(as.matrix(data$data), distance = pv$distance, ef = pv$ef_construction, M = pv$M)
  list(ann = ann, target = data$target, par_vals = pv)
}

#' @export
predictLearner.classif.RcppHNSW = function(.learner, .model, .newdata, ...) {
  nns = RcppHNSW::hnsw_search(as.matrix(.newdata), .model$learner.model$ann,
    ...)
  tgt = .model$learner.model$target
  # Convert to probability matrix
  if (.learner$predict.type == "prob") {
    d = do.call("rbind", lapply(seq_len(nrow(nns$idx)), function(i) {
    prop.table(table(factor(tgt[nns$idx[i,]],
      levels = levels(tgt))))
    }))
  } else {
    d = as.factor(sapply(seq_len(nrow(nns$idx)), function(i) {
      d = tgt[nns$idx[i,]]
      names(which.max(table(d)))
    }))
  }
  return(d)
}
