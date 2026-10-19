def static_threshold(selectivity, density, threshold):
    return float(threshold)


def selectivity_threshold(selectivity, density, low, high):
    return float(high - (high - low) * selectivity)


def density_threshold(selectivity, density, low, high):
    return float(high - (high - low) * density)


def inverted_threshold(selectivity, density, low, high):
    return float(low + (high - low) * selectivity)
