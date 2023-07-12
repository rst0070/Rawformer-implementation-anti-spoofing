
class SysConfig:
    
    def __init__(self):
        
        self.wandb_disabled             = False
        self.wandb_project              = 'ASV-Spoofing'
        self.wandb_name                 = 'SE-Rawformer, no DA'
        self.wandb_entity               = 'rst0070'
        self.wandb_key                  = '8c8d77ae7f92de2b007ad093af722aaae5f31003'
        self.wandb_notes                = 'lr=8*1e-4, ts_hidden=660, rand_seed=1024, pre-emphasis=0.97'
        
        self.path_label_asv_spoof_2019_la_train     = '/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
        self.path_label_asv_spoof_2019_la_dev       = '/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        self.path_asv_spoof_2019_la_train           = '/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac'
        self.path_asv_spoof_2019_la_dev             = '/data/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac'
        
        self.path_label_asv_spoof_2021_la_eval      = '/data/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt'
        self.path_asv_spoof_2021_la_eval      = '/data/ASVspoof2021_LA_eval/flac'
        
        self.num_workers                = 4
        
class ExpConfig:
    
    def __init__(self):
        
        self.random_seed                = 1024
        
        self.pre_emphasis               = 0.97
        
        self.sample_rate                = 16000
        self.train_duration_sec         = 4
        self.test_duration_sec          = 4
        
        self.batch_size_train           = 32
        self.batch_size_test            = 40
        self.embedding_size             = 64
        self.max_epoch                  = 300
        
        self.lr                         = 8 * 1e-4
        self.lr_min                     = 1e-6 # this could not work because i turned off scheduler in some cases
        
        self.transformer_hidden         = 660
        
        self.allow_data_augmentation    = False
        self.data_augmentation          = ['ACN']# additive colored noise    