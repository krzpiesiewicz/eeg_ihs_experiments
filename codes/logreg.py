from itertools import repeat, chain

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split

from .selection import (
    print_res,
    print_res_head,
    print_newlines,
    printed_names,
    reset_res_key,
    add_to_res_key,
    get_res,
    reset_res,
    add_to_res,
    simple_scorings,
)


def logreg_scores(
    fsets,
    fsets_names,
    classes,
    eval_idx="all",
    cv=10,
    model_random_state=42,
    cv_random_state=42,
    scorings_dct=simple_scorings,
    print_scores=True,
    newlines_after_cv=0,
    newlines_after_fset=0,
    return_res=True,
    res_type="dict",  # or "list"
    custom_key=None,
    fset_in_key=False,
):
    vs_for_cv = cv if type(cv) is list else [cv]
    model_random_states = (
        model_random_state if type(model_random_state) is list else [model_random_state]
    )
    if type(fsets) is not list:
        fsets = [fsets]
    if type(eval_idx) is str and eval_idx == "all":
        eval_idx = fsets[0].index
    if type(fsets_names) is not list:
        fsets_names = [fsets_names]
    assert custom_key is not None or fset_in_key

    fsets_print_names = printed_names(fsets_names)

    reset_res(res_type)
    for cv_folds in vs_for_cv:
        model_name = f"Logistic Regression"
        if print_scores:
            model_dct = {}
            model_dct["model_name"] = model_name
            if len(vs_for_cv) > 1:
                model_dct["cv_folds"] = cv_folds
            else:
                model_dct["random_splits"] = len(model_random_states)
            print_res_head(model_dct)
        for fset, fset_print_name, fset_name in zip(
            fsets, fsets_print_names, fsets_names
        ):
            reset_res_key()
            if custom_key is not None:
                add_to_res_key(custom_key)
            if fset_in_key:
                add_to_res_key("fset", fset_name)

            X = fset.loc[
                eval_idx,
            ]
            y = pd.Series(classes[eval_idx], index=eval_idx)
            scores_res = {}

            def train_eval_add_scores(train_idx, test_idx, model_random_state):
                train_idx = eval_idx[train_idx]
                test_idx = eval_idx[test_idx]
                model = LogisticRegression(
                    random_state=model_random_state, max_iter=1000, solver="lbfgs"
                )
                model.fit(
                    X.loc[
                        train_idx,
                    ],
                    y[train_idx],
                )
                for idx, prefix in [(train_idx, "train_"), (test_idx, "test_")]:
                    y_pred = model.predict(
                        X.loc[
                            idx,
                        ]
                    )
                    y_true = y[idx]
                    for score_name, score_fun in scorings_dct.items():
                        key = f"{prefix}{score_name}"
                        score = score_fun(y_true, y_pred)
                        if key not in scores_res:
                            scores_res[key] = score
                        else:
                            scores_res[key] += score

            if cv_folds > 1:
                for (train_idx, test_idx), model_random_state in zip(
                    StratifiedKFold(
                        n_splits=cv_folds,
                        random_state=cv_random_state,
                        shuffle=True,
                    ).split(X, y),
                    model_random_states,
                ):
                    train_eval_add_scores(train_idx, test_idx, model_random_state)
                scores_res = {
                    key: (score / cv_folds) for key, score in scores_res.items()
                }
            else:
                for model_random_state in model_random_states:
                    train_idx, test_idx, _, _ = train_test_split(
                        np.arange(0, len(y)),
                        y,
                        random_state=model_random_state,
                        train_size=0.7,
                    )
                    train_eval_add_scores(train_idx, test_idx, model_random_state)
                scores_res = {
                    key: score / len(model_random_states)
                    for key, score in scores_res.items()
                }
            res_clf = {}
            for score_name in scorings_dct.keys():
                for prefix in ("train_", "test_"):
                    key = f"{prefix}{score_name}"
                    score = scores_res[key]
                    res_clf[key] = score
            res_clf["fset"] = fset_name
            res_clf["fset_print_name"] = fset_print_name
            res_clf["model_name"] = model_name
            if len(vs_for_cv) > 1:
                res_clf["cv_folds"] = cv_folds
            else:
                res_clf["random_splits"] = len(model_random_states)

            if print_scores:
                print_res(res_clf, print_head=False)
            add_to_res(res_clf)
            #                     return return_dct
            print_newlines(newlines_after_fset)
        print_newlines(newlines_after_cv)

    if return_res:
        returned_res = get_res()
        reset_res()
        return returned_res
