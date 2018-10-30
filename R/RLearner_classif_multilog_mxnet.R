#' @export
makeRLearner.classif.mx.multinomial_logistic = function() {
  makeRLearnerClassif(
    cl = "classif.mx.multinomiallogistic",
    package = "mxnet",
    par.set = makeParamSet(
      # other hyperparameters
      makeNumericLearnerParam(id = "validation.ratio"),
      makeIntegerLearnerParam(id = "early.stop.badsteps", lower = 1),
      makeLogicalLearnerParam(id = "early.stop.maximize", default = TRUE),
      makeNumericLearnerParam(id = "dropout.input", lower = 0, upper = 1 - 1e-7),
      makeLogicalLearnerParam(id = "batch.normalization", default = FALSE),
      makeUntypedLearnerParam(id = "ctx", default = mxnet::mx.ctx.default(), tunable = FALSE),
      makeIntegerLearnerParam(id = "begin.round", default = 1L),
      makeIntegerLearnerParam(id = "num.round", default = 10L),
      makeDiscreteLearnerParam(id = "optimizer", default = "sgd",
        values = c("sgd", "rmsprop", "adam", "adagrad", "adadelta")),
      makeUntypedLearnerParam(id = "initializer", default = NULL),
      makeUntypedLearnerParam(id = "eval.data", default = NULL, tunable = FALSE),
      makeUntypedLearnerParam(id = "eval.metric", default = NULL, tunable = FALSE),
      makeUntypedLearnerParam(id = "epoch.end.callback", default = NULL, tunable = FALSE),
      makeUntypedLearnerParam(id = "batch.end.callback", default = NULL, tunable = FALSE),
      makeIntegerLearnerParam(id = "array.batch.size", default = 128L),
      makeDiscreteLearnerParam(id = "array.layout", default = "rowmajor",
        values = c("auto", "colmajor", "rowmajor"), tunable = FALSE),
      makeUntypedLearnerParam(id = "kvstore", default = "local", tunable = FALSE),
      makeLogicalLearnerParam(id = "verbose", default = FALSE, tunable = FALSE),
      makeUntypedLearnerParam(id = "arg.params", tunable = FALSE),
      makeUntypedLearnerParam(id = "aux.params", tunable = FALSE),
      makeUntypedLearnerParam(id = "symbol", tunable = FALSE),
      # optimizer specific hyperhyperparameters
      makeNumericLearnerParam(id = "rho", default = 0.9, requires = quote(optimizer == "adadelta")),
      makeNumericLearnerParam(id = "epsilon",
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam"))),
      makeNumericLearnerParam(id = "wd", default = 0,
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "rescale.grad", default = 1,
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "clip_gradient",
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam", "rmsprop", "sgd"))),
      makeFunctionLearnerParam(id = "lr_scheduler",
        requires = quote(optimizer %in% c("adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "learning.rate",
        requires = quote(optimizer %in% c("adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "beta1", default = 0.9, requires = quote(optimizer == "adam")),
      makeNumericLearnerParam(id = "beta2", default = 0.999, requires = quote(optimizer == "adam")),
      makeNumericLearnerParam(id = "gamma1", default = 0.95,
        requires = quote(optimizer == "rmsprop")),
      makeNumericLearnerParam(id = "gamma2", default = 0.9,
        requires = quote(optimizer == "rmsprop")),
      makeNumericLearnerParam(id = "momentum", default = 0, requires = quote(optimizer == "sgd"))
    ),
    properties = c("twoclass", "multiclass", "numerics", "prob"),
    par.vals = list(learning.rate = 0.1, array.layout = "rowmajor", verbose = FALSE),
    name = "Multinomial Logistic Regression",
    short.name = "mxml",
    note = "Default of `learning.rate` set to `0.1`.
      Default of `array.layout` set to `'rowmajor'`.
      Default of `verbose` is set to `FALSE`.
      Default of `initializer` is set to NULL, which results in the default mxnet initializer being called when
      training a model. Number of output nodes is detected automatically.
      The upper bound for dropout is set to `1 - 1e-7` as in `mx.mlp`
      in the `mxnet` package."
  )
}

#' @export
trainLearner.classif.mx.multinomiallogistic = function(.learner, .task, .subset, .weights = NULL,
  dropout.input = NULL, batch.normalization = FALSE, validation.ratio = NULL,
  eval.data = NULL, early.stop.badsteps = NULL, epoch.end.callback = NULL,
  early.stop.maximize = TRUE, array.layout = "rowmajor", ...) {

  # transform data in correct format
  d = getTaskData(.task, subset = .subset, target.extra = TRUE)
  y = as.numeric(d$target) - 1
  x = data.matrix(d$data)

  # construct validation data in case validation.ratio > 0
  if (is.null(eval.data) & !is.null(validation.ratio)) {
    eval.data = list()
    rdesc = makeResampleDesc("Holdout", split = 1 - validation.ratio, stratify = TRUE)
    rinst = makeResampleInstance(rdesc, subsetTask(.task, subset = .subset))
    val.ind = rinst$test.inds[[1]]
    eval.data$label = y[val.ind]
    y = y[-val.ind]
    eval.data$data = x[val.ind, ]
    x = x[-val.ind, ]
  }

  # early stopping
  if (is.null(epoch.end.callback) & !is.null(early.stop.badsteps)) {
    epoch.end.callback = mxnet::mx.callback.early.stop(bad.steps = early.stop.badsteps,
      maximize = early.stop.maximize)
  }

  # construct vectors with #nodes and activations

  sym = mxnet::mx.symbol.Variable("data")
  # Input dropout
  if (!is.null(dropout.input))
    sym = mxnet::mx.symbol.Dropout(sym, p = dropout.input)
  if (!is.null(batch.normalization))
    sym = mxnet::mx.symbol.BatchNorm(sym)

  sym = mxnet::mx.symbol.FullyConnected(sym, num_hidden = nlevels(d$target))
  out = mxnet::mx.symbol.SoftmaxOutput(sym)

  # create model
  model = mxnet::mx.model.FeedForward.create(out, X = x, y = y, eval.data = eval.data,
    epoch.end.callback = epoch.end.callback, array.layout = array.layout, ...)
  return(model)
}

#' @export
predictLearner.classif.mx.multinomiallogistic = function(.learner, .model, .newdata, ...) {
  x = data.matrix(.newdata)

  p = predict(.model$learner.model, X = x, array.layout = .model$learner$par.vals$array.layout)
  if (.learner$predict.type == "response") {
    # in very rare cases, the mxnet FeedForward algorithm does not converge and returns useless /
    # error output in the probability matrix. In this case, which.max returns integer(0).
    # To avoid errors, return NA instead.
    p = apply(p, 2, function(i) {
      w = which.max(i)
      return(ifelse(length(w > 0), w, NA))
    })
    p = factor(p, exclude = NA)
    levels(p) = .model$task.desc$class.levels
    return(p)
  }
  if (.learner$predict.type == "prob") {
    p = t(p)
    colnames(p) = .model$task.desc$class.levels
    return(p)
  }
}

