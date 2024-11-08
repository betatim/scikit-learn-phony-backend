import importlib


def get_implementation(estimator_name):
    """Get the backend for an estimator.

    The name passed to this function will be the full name of the estimator
    that is being dispatched. It will include modules and sub-modules.

    If `can_has` returned True then this function has to return a backend.
    """
    # Remove the leading `sklearn.`
    _, estimator_name = estimator_name.split(".", maxsplit=1)
    module_name, estimator_name = estimator_name.rsplit(":", maxsplit=1)

    mod = importlib.import_module(
        "sklearn_phony_backend.implementations." + module_name
    )

    klass = getattr(mod, estimator_name)
    return klass


def can_has(estimator_name, estimator, *fit_args, **fit_kwargs):
    """Evaluate if the backend wants to take this call or not.

    The arguments are the full name of the estimator being dispatched, the
    estimator instance as well as the arguments the user provided to the fit
    method.

    This function should return quickly if the decision is negative.
    """
    if estimator_name != "sklearn.cluster:KMeans":
        return False

    if estimator.init != "random":
        return False

    return True