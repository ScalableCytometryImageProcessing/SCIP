from typing import Mapping

from scip.utils.util import copy_without
import dask.bag
import scipy.stats


def feature_partition(part):
    for i in range(len(part)):
        if "pixels" in part[i]:
            part[i]["filter_sum"] = part[i]["pixels"][0].sum()
    return part


def item(bag: dask.bag.Bag) -> Mapping[str, dask.bag.Item]:
    bag = bag.filter(lambda a: "filter_sum" in a).map(lambda a: a["filter_sum"])
    mu = bag.mean()
    std = bag.std()
    return dict(mu=mu, std=std)


def predicate(x, *, mu, std):
    q5 = scipy.stats.norm.ppf(0.05, loc=mu, scale=std)
    if ("filter_sum" in x) and (x["filter_sum"] > q5):
        return x
    else:
        return copy_without(x, without=["mask", "pixels"])
