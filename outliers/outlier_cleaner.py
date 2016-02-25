#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    from operator import itemgetter # sort by custom index in tuple

    errors = sorted([(at, nwt, abs(float(nwt-p))) for at, nwt, p in zip(ages, net_worths, predictions)], key=itemgetter(2))
    cleaned = errors[:int(len(errors)-(len(errors)*0.1))]

    return cleaned
