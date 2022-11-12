class KinematicsConfig:

    configs = {
      # possible methods: normal, viot, prob, rand
      'method' : 'prob',
      'step_size' : 0.1,
      'num_samples' : 10,
      'vars' : [0.5, 1.5, 3],
      'aug_vars' : 0.8,
      'std_thresh' : 0.5,
      'poly_degree' : 2,
      'bspline_degree' : 3,
      'pos_buff_size' : 40,
      # possible methods: proportionality, direct
      'range_estimation_method' : 'direct'
    }
