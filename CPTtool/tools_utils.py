"""
Tools for the cpt module
"""


def n_iter(n, qt, friction_nb, sigma_eff, sigma_tot, Pa):
    """
    Computation of stress exponent *n*

    :param n: initial stress exponent
    :param qt: tip resistance
    :param friction_nb: friction number
    :param sigma_eff: effective stress
    :param sigma_tot: total stress
    :param Pa: atmospheric pressure
    :return: updated n - stress exponent
    """
    import numpy as np

    # convergence of n
    Cn = (Pa / np.array(sigma_eff)) ** n

    Q = ((np.array(qt) - np.array(sigma_tot)) / Pa) * Cn
    F = (np.array(friction_nb) / (np.array(qt) - np.array(sigma_tot))) * 100

    # Q and F cannot be negative. if negative, log10 will be infinite.
    # These values are limited by the contours of soil behaviour of Robertson
    Q[Q <= 1.] = 1.
    F[F <= 0.1] = 0.1
    Q[Q >= 1000.] = 1000.
    F[F >= 10.] = 10.

    IC = ((3.47 - np.log10(Q)) ** 2. + (np.log10(F) + 1.22) ** 2.) ** 0.5

    n = 0.381 * IC + 0.05 * (sigma_eff / Pa) - 0.15
    n[n > 1.] = 1.
    return n


def merge(min_thick, depth, lithology):
    """
    Merge of the CPT layers into a minimum thickness

    :param min_thick: minimum layer thickness
    :param depth: CPT depth
    :param lithology: layer lithology
    :return: thickness, depth, lithology, indices
    """
    import numpy as np

    # find location of the start of the soil types
    aux = ""
    idx = []
    # here the i is where I am in the Cpt , val is the value of the lithology which is a number coresponding to Robertson
    # if the previous layer is not the same as the next save the position and save the value for the next step
    for i, val in enumerate(lithology):
        if val != aux:
            aux = val
            idx.append(i)
    # retrieve the depths of the indexes found
    # z_ini - depth at the start of new layer
    z_ini = [depth[i] for i in idx]
    # thickness between those layers calculated
    thick = np.diff(z_ini)
    # soil type
    label = [lithology[i] for i in idx]

    # merge the depths
    try:
        id = np.where(np.array(thick) <= min_thick)[0][0]
    except IndexError:
        return thick, z_ini, label

    del z_ini[id + 1]
    # determine the biggest label
    if thick[id] >= thick[id + 1]:
        del label[id + 1]
    else:
        del label[id]

    thick = np.diff(z_ini)
    return thick, z_ini, label


def compute_probability(coord_cpt, coord_src, coord_rec):
    r"""
    Compute the probability for each scenario following the following formula:

    .. math::

        w_{i} = \frac{1 - \frac{R_{S,i} \cdot \left(R_{S,i} + R_{R,i} \right)}
                               {\sum_{i=1}^{n}R_{S,i} \cdot \left(R_{S,i} + R_{R,i} \right)}}{n - 1}


    where :math:`w_i` is the probability associated with the *i* scenario, :math:`R_{S,i}` the distance between the CPT
    and the source location, :math:`R_{R,i}` the distance between the CPT and the receiver location, and *n* is the
    number of scenarios/CPTs.

    :param coord_cpt: list of coordinates of the CPTs
    :param coord_src: coordinates of the source
    :param coord_rec: coordinates of the receiver
    :return: probs: probability of occurrence of the scenario
    """
    import numpy as np

    # number of scenarios
    nb_scenarios = len(coord_cpt)

    distance_to_source = []
    distance_to_receiver = []

    # if there if only one scenario: prob = 100
    if nb_scenarios == 1:
        return [100.]

    # iterate around json
    for i in range(nb_scenarios):
        distance_to_source.append(np.sqrt((coord_cpt[i][0] - coord_src[0])**2 +
                                          (coord_cpt[i][1] - coord_src[1])**2))
        distance_to_receiver.append(np.sqrt((coord_cpt[i][0] - coord_rec[0])**2 +
                                            (coord_cpt[i][1] - coord_rec[1])**2))

    sum_weights = np.sum([distance_to_source[i] * (distance_to_source[i] + distance_to_receiver[i]) for i in range(nb_scenarios)])

    probs = []
    for i in range(nb_scenarios):
        weight = (1 - (distance_to_source[i] * (distance_to_source[i] + distance_to_receiver[i])) / sum_weights) / (nb_scenarios - 1)
        probs.append(weight * 100.)

    return probs
