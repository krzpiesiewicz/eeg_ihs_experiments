import IPython.display
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)


progress = 0
total_work = 0

start_time = None
end_time = None


def start_timer():
    global start_time, endtime
    start_time = datetime.now()
    endtime = start_time


def set_work(total_work_count):
    global progress, total_work
    progress = 0
    total_work = total_work_count


def print_progress(change=0, end=""):
    global progress, total_work, start_time, endtime
    progress += change
    end_time = datetime.now()
    tdelta = end_time - start_time
    tdelta = tdelta - timedelta(microseconds=tdelta.microseconds)
    print(
        f"\rprogress {round(100 * progress / total_work, 2)}% – "
        + "Duration: {}        ".format(tdelta),
        end=end,
    )


default_labels = ["Ictal", "Interictal", "Normal"]


def confusion_score(y_true, y_pred, labels=default_labels, normalize="true", **kwargs):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize, **kwargs)
    cm = pd.DataFrame(cm).set_index(
        pd.MultiIndex.from_tuples(
            list(zip(["True"] * len(default_labels), default_labels))
        )
    )
    cm.columns = pd.MultiIndex.from_tuples(
        list(zip(["Predicted"] * len(default_labels), default_labels))
    )
    return cm


simple_scorings = {
    "acc": accuracy_score,
    "bacc": balanced_accuracy_score,
}

full_scorings = simple_scorings.copy()
full_scorings["confusion"] = confusion_score


def print_res_head(res_dct):
    print_buff = ""
    if "model_name" in res_dct:
        print_buff += res_dct["model_name"]
    if "cv_folds" in res_dct:
        print_buff += f', cv_folds: {res_dct["cv_folds"]}'
    if "random_splits" in res_dct:
        print_buff += f', random_splits: {res_dct["random_splits"]}'
    print(print_buff)


def print_res(
    results,
    nondisplayed_metrics=["confusion"],
    print_head=True,
    print_only_first=False,
    new_line=False,
    new_line_between=False,
    train_confusion_matrix=False,
    test_confusion_matrix=False,
):
    if type(results) is list:
        res_lst = results
    else:
        res_lst = [results]
    is_first = True
    for elem in res_lst:
        if type(elem) is tuple:
            res_dct = elem[1]
        else:
            res_dct = elem
        if print_res_head and (is_first or not print_only_first):
            is_first = False
            print_res_head(res_dct)
        print_buff = "{} {} ".format(
            res_dct["fset_print_name"], "-" * (32 - len(res_dct["fset_print_name"]))
        )
        sep = ""
        for prefix in ("train_", "test_"):
            for score_name, score_res in res_dct.items():
                if score_name.startswith(prefix):
                    if all(
                        [substr not in score_name for substr in nondisplayed_metrics]
                    ):
                        print_buff += "{}{}: {:.2%}".format(sep, score_name, score_res)
                        sep = ", "
        print(print_buff)
        if new_line_between:
            print("")
        if train_confusion_matrix:
            print(f"train confusion matrix:")
            display(res_dct[f"train_confusion"])
        if test_confusion_matrix:
            print(f"test confusion matrix:")
            display(res_dct[f"test_confusion"])
    if new_line:
        print("")
        
        
def printed_names(names):
    L = max([len(name) for name in names])
    return [name + " " * (L - len(name)) for name in names]


def print_newlines(c):
    for _ in range(c):
        print("")


res_key = None


def get_key():
    global res_key
    return res_key


def reset_res_key():
    global res_key
    res_key = None


def add_to_res_key(name, val=None):
    global res_key
    if val is None:
        if type(name) is tuple:
            key = name
        else:
            key = (name,)
    else:
        key = (name, val)
    if res_key is None:
        res_key = key
    else:
        res_key = res_key + key


res = None
res_type = None


def get_res():
    global res
    return res


def reset_res(typ="dict"):  # or "list"
    global res, res_type
    res_type = typ
    if typ == "dict":
        res = {}
    if typ == "list":
        res = []


def add_to_res(val):
    global res, res_key, res_type
    if res_type == "dict":
        res[res_key] = val
    if res_type == "list":
        res.append(val)


def sorted_list_by_first(scores_lst):
    return sorted(scores_lst, key=lambda t: t[0], reverse=True)


def sorted_by_score(res_dct_or_list, scores_names, params_names="all"):
    scores_lst = []
    res_lst = None
    if type(res_dct_or_list) is dict:
        res_lst = [dct for _, dct in res_dct_or_list.items()]
    else:
        res_lst = [tupl[1] for tupl in res_dct_or_list]

    for res in res_lst:
        if type(params_names) is str and params_names == "all":
            selected_res = res.copy()
        else:
            selected_res = {}
            for key in params_names:
                selected_res[key] = res[key]
        score_key = []
        for score_name in scores_names:
            score_key.append(round(res[score_name], 4))
        scores_lst.append((tuple(score_key), selected_res))
    return sorted_list_by_first(scores_lst)
