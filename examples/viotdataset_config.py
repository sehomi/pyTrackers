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
        "park_mavic_1":[1,1005],
        # # "park_mavic_1":[1,200],
        # # "park_mavic_1":[110,190],
        # # "park_mavic_1":[1,630],
        # # "park_mavic_2":[45,945],
        "park_mavic_2":[1,945],
        "park_mavic_3":[1,2022],
        # # "park_mavic_3":[710,1100],
        "park_mavic_4":[1,1906],
        # # "park_mavic_4":[1,500],
        # # "park_mavic_5":[840,1697],
        "park_mavic_5":[840,1697],
        "park_mavic_6":[1,1137],
        "park_mavic_7":[1,915],
        # # "park_mavic_7":[1,360],
        # # "park_mavic_7":[360,915],
        "soccerfield_mavic_3":[1,1104],
        # # "soccerfield_mavic_3":[1,500],
        # # "soccerfield_mavic_4":[1,1297]
        "soccerfield_mavic_4":[500,1297]
    }

    params={
        "KCF_HOG":[0.1, 1],
        "ECO":[0.5, 1],
        "DIMP50":[0.3, 1],
        "PRDIMP50":[0.3, 1],
        "KYS":[0.2, 1],
        "TOMP":[0.5, 1],
        "CSRDCF":[0.2, 1],
        "LDES":[0.3, 1],
        "STRCF":[0.2, 1]
    }
