from operator import attrgetter


def selBestWODuplicates(individuals, k, fit_attr="fitness"):
    """Select the *k* best individuals among the input *individuals*. The
    list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list containing the k best individuals.
    """
    rpr = []
    wo_duplicates = []
    for ind in individuals:
        if ind.__repr__() not in rpr:
            rpr.append(ind.__repr__())
            wo_duplicates.append(ind)
    return sorted(wo_duplicates, key=attrgetter(fit_attr), reverse=True)[:k]