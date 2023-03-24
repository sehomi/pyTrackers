class VIOTDatasetConfig:

    fov={
        "cup_0.5HZ":55.0,
        "cup_0.9HZ":55.0,
        "cup_1.1HZ":55.0,
        "cup_1.5HZ":55.0,
        "cup_1.8HZ":55.0,
        "cup_2.1HZ":55.0,
        "cup_3.2HZ":55.0,
        "park_mavic_1":66.0,
        "park_mavic_2":66.0,
        "park_mavic_3":66.0,
        "park_mavic_4":66.0,
        "park_mavic_5":66.0,
        "park_mavic_6":66.0,
        "park_mavic_7":66.0,
        "soccerfield_mavic_3":66.0,
        "soccerfield_mavic_4":66.0
    }

    frames={
        # "cup_0.5HZ":[1,220],
        # "cup_0.9HZ":[1,760],
        # "cup_1.1HZ":[1,329],
        # "cup_1.5HZ":[1,312],
        # "cup_1.8HZ":[1,357],
        # "cup_2.1HZ":[1,465],
        # "cup_3.2HZ":[1,254],
        "park_mavic_1":[110,191],
        # "park_mavic_1":[1,1005],
        # "park_mavic_2":[45,945],
        # "park_mavic_3":[710,1100],
        # "park_mavic_4":[1,500],
        # "park_mavic_5":[840,1697],
        # "park_mavic_6":[1,1137],
        # "park_mavic_7":[1,360],
        # "soccerfield_mavic_3":[1,500],
        # "soccerfield_mavic_4":[500,1297]
    }

    params={
        "KCF_HOG":{"cup_0.5HZ":[0.1, 0], "cup_0.9HZ":[0.1, 0], "cup_1.1HZ":[0.1, 0], \
                   "cup_1.5HZ":[0.1, 0], "cup_1.8HZ":[0.1, 0], "cup_2.1HZ":[0.1, 0], \
                   "cup_3.2HZ":[0.1, 0], "park_mavic_1":[0.1, 1], "park_mavic_2":[0.1, 1], \
                   "park_mavic_3":[0.1, 1], "park_mavic_4":[0.1, 1], "park_mavic_5":[0.1, 1], \
                   "park_mavic_6":[0.1, 1], "park_mavic_7":[0.1, 1], "soccerfield_mavic_3":[0.1, 1], \
                   "soccerfield_mavic_4":[0.1, 1]},
        "ECO":{"cup_0.5HZ":[0.5, 0], "cup_0.9HZ":[0.5, 0], "cup_1.1HZ":[0.5, 0], \
               "cup_1.5HZ":[0.5, 0], "cup_1.8HZ":[0.5, 0], "cup_2.1HZ":[0.5, 0], \
               "cup_3.2HZ":[0.5, 0], "park_mavic_1":[0.5, 1], "park_mavic_2":[0.5, 1], \
               "park_mavic_3":[0.5, 1], "park_mavic_4":[0.5, 1], "park_mavic_5":[0.5, 1], \
               "park_mavic_6":[0.5, 1], "park_mavic_7":[0.5, 1], "soccerfield_mavic_3":[0.5, 1], \
               "soccerfield_mavic_4":[0.5, 1]},
        "DIMP50":{"cup_0.5HZ":[0.2, 1.1], "cup_0.9HZ":[0.2, 1.1], "cup_1.1HZ":[0.2, 1.1], \
                  "cup_1.5HZ":[0.2, 1.1], "cup_1.8HZ":[0.2, 1.1], "cup_2.1HZ":[0.2, 1.1], \
                  "cup_3.2HZ":[0.2, 1.1], "park_mavic_1":[0.2, 1.1], "park_mavic_2":[0.2, 1.1], \
                  "park_mavic_3":[0.2, 1.1], "park_mavic_4":[0.2, 1.1], "park_mavic_5":[0.2, 1.1], \
                  "park_mavic_6":[0.2, 1.1], "park_mavic_7":[0.2, 1.1], "soccerfield_mavic_3":[0.2, 1.1], \
                  "soccerfield_mavic_4":[0.2, 1.1]},
        "PRDIMP50":{"cup_0.5HZ":[0.3, 1], "cup_0.9HZ":[0.3, 1], "cup_1.1HZ":[0.3, 1], \
                  "cup_1.5HZ":[0.3, 1], "cup_1.8HZ":[0.3, 1], "cup_2.1HZ":[0.3, 1], \
                  "cup_3.2HZ":[0.3, 1], "park_mavic_1":[0.3, 1], "park_mavic_2":[0.3, 1], \
                  "park_mavic_3":[0.3, 1], "park_mavic_4":[0.3, 1], "park_mavic_5":[0.3, 1], \
                  "park_mavic_6":[0.3, 1], "park_mavic_7":[0.3, 1], "soccerfield_mavic_3":[0.3, 1], \
                  "soccerfield_mavic_4":[0.3, 1]},
        "KYS":{"cup_0.5HZ":[0.2, 1.1], "cup_0.9HZ":[0.2, 1.1], "cup_1.1HZ":[0.2, 1.1], \
                  "cup_1.5HZ":[0.2, 1.1], "cup_1.8HZ":[0.2, 1.1], "cup_2.1HZ":[0.2, 1.1], \
                  "cup_3.2HZ":[0.2, 1.1], "park_mavic_1":[0.2, 1.1], "park_mavic_2":[0.2, 1.1], \
                  "park_mavic_3":[0.2, 1.1], "park_mavic_4":[0.2, 1.1], "park_mavic_5":[0.2, 1.1], \
                  "park_mavic_6":[0.2, 1.1], "park_mavic_7":[0.2, 1.1], "soccerfield_mavic_3":[0.2, 1.1], \
                  "soccerfield_mavic_4":[0.2, 1.1]},
        "TOMP":{"cup_0.5HZ":[0.5, 1], "cup_0.9HZ":[0.5, 1], "cup_1.1HZ":[0.5, 1], \
                  "cup_1.5HZ":[0.5, 1], "cup_1.8HZ":[0.5, 1], "cup_2.1HZ":[0.5, 1], \
                  "cup_3.2HZ":[0.5, 1], "park_mavic_1":[0.5, 1], "park_mavic_2":[0.5, 1], \
                  "park_mavic_3":[0.5, 1], "park_mavic_4":[0.5, 1], "park_mavic_5":[0.5, 1], \
                  "park_mavic_6":[0.5, 1], "park_mavic_7":[0.5, 1], "soccerfield_mavic_3":[0.5, 1], \
                  "soccerfield_mavic_4":[0.5, 1]},
        "MIXFORMER_VIT":{"cup_0.5HZ":[0.995, 0], "cup_0.9HZ":[0.995, 0], "cup_1.1HZ":[0.995, 0], \
                 "cup_1.5HZ":[0.995, 0], "cup_1.8HZ":[0.995, 0], "cup_2.1HZ":[0.995, 0], \
                 "cup_3.2HZ":[0.995, 0], "park_mavic_1":[0.995, 1], "park_mavic_2":[0.995, 1], \
                 "park_mavic_3":[0.995, 1], "park_mavic_4":[0.995, 1], "park_mavic_5":[0.995, 1], \
                 "park_mavic_6":[0.995, 1], "park_mavic_7":[0.995, 1], "soccerfield_mavic_3":[0.995, 1], \
                 "soccerfield_mavic_4":[0.995, 1]},
        "CSRDCF":{"cup_0.5HZ":[0.2, 0], "cup_0.9HZ":[0.2, 0], "cup_1.1HZ":[0.2, 0], \
                  "cup_1.5HZ":[0.2, 0], "cup_1.8HZ":[0.2, 0], "cup_2.1HZ":[0.2, 0], \
                  "cup_3.2HZ":[0.2, 0], "park_mavic_1":[0.2, 1], "park_mavic_2":[0.2, 1], \
                  "park_mavic_3":[0.2, 1], "park_mavic_4":[0.2, 1], "park_mavic_5":[0.2, 1], \
                  "park_mavic_6":[0.2, 1], "park_mavic_7":[0.2, 1], "soccerfield_mavic_3":[0.2, 1], \
                  "soccerfield_mavic_4":[0.2, 1]},
        "LDES":{"cup_0.5HZ":[0.3, 0], "cup_0.9HZ":[0.3, 0], "cup_1.1HZ":[0.3, 0], \
                "cup_1.5HZ":[0.3, 0], "cup_1.8HZ":[0.3, 0], "cup_2.1HZ":[0.3, 0], \
                "cup_3.2HZ":[0.3, 0], "park_mavic_1":[0.3, 1], "park_mavic_2":[0.3, 1], \
                "park_mavic_3":[0.3, 1], "park_mavic_4":[0.3, 1], "park_mavic_5":[0.3, 1], \
                "park_mavic_6":[0.3, 1], "park_mavic_7":[0.3, 1], "soccerfield_mavic_3":[0.3, 1], \
                "soccerfield_mavic_4":[0.3, 1]},
        "STRCF":{"cup_0.5HZ":[0.2, 0], "cup_0.9HZ":[0.2, 0], "cup_1.1HZ":[0.2, 0], \
                 "cup_1.5HZ":[0.2, 0], "cup_1.8HZ":[0.2, 0], "cup_2.1HZ":[0.2, 0], \
                 "cup_3.2HZ":[0.2, 0], "park_mavic_1":[0.2, 1], "park_mavic_2":[0.2, 1], \
                 "park_mavic_3":[0.2, 1], "park_mavic_4":[0.2, 1], "park_mavic_5":[0.2, 1], \
                 "park_mavic_6":[0.2, 1], "park_mavic_7":[0.2, 1], "soccerfield_mavic_3":[0.2, 1], \
                 "soccerfield_mavic_4":[0.2, 1]},
    }
