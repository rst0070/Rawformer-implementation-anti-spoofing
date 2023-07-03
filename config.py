
class SysConfig:
    
    def __init__(self):
        
        self.wandb_disabled             = False
        self.wandb_entity               = 'rst0070'
        self.wandb_key                  = '8c8d77ae7f92de2b007ad093af722aaae5f31003'
        
        self.path_label_asv_spoof_2019_la_train     = '/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
        self.path_label_asv_spoof_2019_la_dev       = '/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        self.path_asv_spoof_2019_la_train           = '/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac'
        self.path_asv_spoof_2019_la_dev             = '/data/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac'
        
        self.path_label_asv_spoof_2021_la_eval      = '/data/ASV_spoof/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt'
        self.path_asv_spoof_2021_la_eval      = '/data/ASV_spoof/ASVspoof2021_LA_eval/flac'
        
class ExpConfig:
    
    def __init__(self):
        
        self.sample_rate                = 16000
        self.train_duration_sec         = 4
        self.test_duration_sec          = 4
        
        self.embedding_size             = 64
        
        self.lr                         = 8 * 1e-4