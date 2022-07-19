class KinematicsConfig:

    configs = {
      # possible methods: normal, viot, prob
      'method' : 'prob',
      'step_size' : 0.1,
      'num_samples' : 10,
      'vars' : [0.5, 1.5, 3],
      'aug_vars' : 0.8,
      'std_thresh' : 0.5,
      'poly_degree' : 1,
      'bspline_degree' : 4,
      'pos_buff_size' : 40
    }