from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/content/pyTrackers/MixFormer/data/got10k_lmdb'
    settings.got10k_path = '/content/pyTrackers/MixFormer/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/content/pyTrackers/MixFormer/data/lasot_lmdb'
    settings.lasot_path = '/content/pyTrackers/MixFormer/data/lasot'
    settings.network_path = '/content/pyTrackers/MixFormer/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/content/pyTrackers/MixFormer/data/nfs'
    settings.otb_path = '/content/pyTrackers/MixFormer/data/OTB2015'
    settings.prj_dir = '/content/pyTrackers/MixFormer'
    settings.result_plot_path = '/content/pyTrackers/MixFormer/test/result_plots'
    settings.results_path = '/content/pyTrackers/MixFormer/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/content/pyTrackers/MixFormer'
    settings.segmentation_path = '/content/pyTrackers/MixFormer/test/segmentation_results'
    settings.tc128_path = '/content/pyTrackers/MixFormer/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/content/pyTrackers/MixFormer/data/trackingNet'
    settings.uav_path = '/content/pyTrackers/MixFormer/data/UAV123'
    settings.vot_path = '/content/pyTrackers/MixFormer/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

