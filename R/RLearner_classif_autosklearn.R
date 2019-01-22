autosklearn_estimators = c("adaboost", "bernoulli", "random_forest", "decision_tree", "lda",
  "extra_trees", "gaussian_nb", "gradient_boosting", "k_nearest_neighbours", "liblinear_svc", "libsvm_svc",
  "multinomlial_nb", "passive_aggressive", "qda", "sgd", "xgradient_boosting")
# Attention: Some preprocessors are missing
autosklearn_preprocessors = c("no_preprocessing", "densifier", "fast_ica")

#' @export
makeRLearner.classif.autosklearn = function() {
  makeRLearnerClassif(
    cl = "classif.autosklearn",
    package = "reticulate",
    # For full paramset see https://automl.github.io/auto-sklearn/master/api.html
    # Attention: Defaults are not exactly as in  autosklearn
    par.set = makeParamSet(
      makeIntegerLearnerParam("time_left_for_this_task", lower = 1L, upper = Inf, default = 3600L),
      makeIntegerLearnerParam("per_run_time_limit", lower = 1L, upper = Inf, default = 360L),
      makeIntegerLearnerParam("initial_configurations_via_metalearning", lower = 0L, upper = Inf, default = 25L),
      makeUntypedLearnerParam("include_estimators", default = list("random_forest"), special.vals = list(NULL)),
      makeUntypedLearnerParam("include_preprocessors", default = list("no_preprocessing"), special.vals = list(NULL)),
      makeUntypedLearnerParam("exclude_estimators", default = NULL, special.vals = list(NULL)),
      makeUntypedLearnerParam("exclude_preprocessors", default = NULL, special.vals = list(NULL)),
      makeIntegerLearnerParam("ensemble_size", lower = 0L, upper = Inf, default = 50L),
      makeIntegerLearnerParam("ensemble_nbest", lower = 0L, upper = Inf, default = 50L)
    ),
    properties = c("twoclass", "multiclass", "numerics", "prob", "missings", "factors"),
    name = "Autosklearn",
    short.name = "autosklearn",
    note = "Defaults deviate from autosklearn defaults"
    )
}

#' @export
trainLearner.classif.autosklearn = function(.learner, .task, .subset, .weights = NULL, ...) {

  autosklearn = import("autosklearn")

  classifier = autosklearn$classification$AutoSklearnClassifier(...)

  train = getTaskData(.task, .subset, target.extra = TRUE)
  feat.type = ifelse(vlapply(train$data, is.factor), "Categorical", "Numerical")

  classifier$fit(as.matrix(train$data), train$target, feat_type = feat.type)
  return(classifier)
}

#' @export
predictLearner.classif.autosklearn = function(.learner, .model, .newdata, ...) {
  as.factor(.model$learner.model$predict(as.matrix(.newdata)))
}
