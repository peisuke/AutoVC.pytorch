class hparams:
    sample_rate = 16000 
    n_fft = 1024
    #fft_bins = n_fft // 2 + 1
    num_mels = 80
    hop_length = 256
    win_length = 1024
    fmin = 90
    fmax = 7600
    min_level_db = -100
    ref_level_db = 20
    
    seq_len_factor = 64
    bits = 12
    seq_len = seq_len_factor * hop_length
    
    dim_neck = 16
    dim_emb = 256
    dim_pitch = 256
    dim_pre = 512
    freq = 32
    
    ## wavenet vocoder
    builder = 'wavenet'
    hop_size = 256
    log_scale_min = float(-32.23619130191664)
    out_channels = 10 * 3
    layers = 24
    stacks = 4
    residual_channels = 512
    gate_channels = 512
    skip_out_channels = 256
    dropout = 1 - 0.95
    kernel_size = 3
    cin_channels = 80
    upsample_conditional_features = True
    upsample_scales = [4, 4, 4, 4]
    freq_axis_kernel_size = 3
    gin_channels = -1
    n_speakers = -1
    weight_normalization = True
    legacy = True
