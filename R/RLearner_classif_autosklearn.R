#' @export
makeRLearner.classif.autosklearn = function() {
  makeRLearnerClassif(
    cl = "classif.autosklearn",
    package = "reticulate",
    par.set = makeParamSet(
      makeIntegerLearnerParam("time_left_for_this_task", lower = 1L, upper = Inf, default = 120L),
      makeIntegerLearnerParam("ensemble_size", lower = 0L, upper = Inf, default = 1L),
      makeIntegerLearnerParam("initial_configurations_via_metalearning", lower = 0L, upper = Inf, default = 1L)
    ),
    properties = c("twoclass", "multiclass", "numerics", "prob", "missings", "factors"),
    name = "Autosklearn",
    short.name = "autosklearn",
    note = ""
    )
}

#' @export
trainLearner.classif.autosklearn = function(.learner, .task, .subset, .weights = NULL, ...) {

  autosklearn = import("autosklearn")

  classifier = autosklearn$classification$AutoSklearnClassifier(
     include_estimators=list("random_forest"), exclude_estimators=NULL,
     include_preprocessors = list("no_preprocessing"), exclude_preprocessors=NULL,
     ...)

  train = getTaskData(.task, .subset, target.extra = TRUE)
  feat.type = ifelse(vlapply(train$data, is.factor), "Categorical", "Numerical")

  classifier$fit(as.matrix(train$data), train$target, feat_type = feat.type)
  return(classifier)
}

#' @export
predictLearner.classif.autosklearn = function(.learner, .model, .newdata, ...) {
  as.factor(.model$learner.model$predict(as.matrix(.newdata)))
}
