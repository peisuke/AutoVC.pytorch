class hparams:
    sample_rate = 16000 
    n_fft = 1024
    fft_bins = n_fft // 2 + 1
    num_mels = 80
    hop_length = 256
    win_length = 1024
    fmin = 40
    min_level_db = -100
    ref_level_db = 20
    #segment_length = 16384
    
    seq_len_factor = 64
    bits = 12
    seq_len = seq_len_factor * hop_length