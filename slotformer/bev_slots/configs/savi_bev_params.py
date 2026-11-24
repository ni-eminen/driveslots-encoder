from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 4
    max_epochs = 30  # ~80k steps
    save_interval = 0.2  # save every 0.2 epoch
    eval_interval = 2  # evaluate every 2 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 5  # visualization after each epoch
    sampler_seed = 42  # regenerate training order, including on checkpoint resume
    loss_spike_root = 'outputs/diagnostics/loss-spikes'
    loss_spike_window = 100
    loss_spike_warmup = 20
    loss_spike_factor = 2.0
    loss_spike_min_delta = 0.05
    loss_spike_cooldown = 100

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-5
    override_resume_lr = False  # restore the checkpoint's LR and scheduler state
    clip_grad = 0.05  # following the paper
    warmup_steps_pct = 0.025  # warmup in the first 2.5% of total steps

    # data settings
    dataset = 'obj3d_for_bev'
    data_root = ''
    crop_bevs = True
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 50  # take the first 50 frames of each video
    train_batch_size = 16 #64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'StoSAVi'
    resolution = (192, 192)
    input_frames = n_sample_frames


    # Slot Attention
    slot_dict = dict(
        num_slots=30,
        slot_size=256,
        slot_mlp_size=256,
        num_iterations=2,
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=256,
        enc_norm='',
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(256, 64, 64, 64, 64),
        dec_resolution=(12, 12),
        dec_ks=[3,3,5,5],
        dec_norm='',
        upscale=True
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=512,
        pred_sg_every=None,
    )

    # loss configs
    loss_dict = dict(
        recon_save_interval=1000,  # training steps; 0 disables image export
        recon_save_root='outputs/diagnostics/reconstructions',
        use_post_recon_loss=True,
        # kld_method='none',  # standard SAVi
        kld_method='var-0.01',  # prior Gaussian variance is 0.01
        sample_slot_noise=True,  # sample using predicted means and variances


        # ---- perceptual reconstruction loss (optional) ----
        # See slotformer/bev_slots/models/perceptual.py and
        # Default 'none' keeps the MSE-only
        # baseline; switching to 'lpips_vgg' or 'feat_l2_robust_r50' turns the
        # per-frame reconstruction loss into  pixel_weight*MSE + perceptual_weight*L_perc.
        perceptual_loss='feat_l2_vgg16_relu2_2',  # 'none' | 'lpips_vgg' | 'feat_l2_robust_r50' | feat_l2_vgg16_relu2_2 | feat_l2_vgg16_relu1_2_relu2_2
        pixel_weight=1.0,
        perceptual_weight=0.1,
        robust_ckpt=None,  # required when perceptual_loss='feat_l2_robust_r50'
        stabilize_kld_loss_logvar=True,  # stabilize log_var in KLD loss (see slotformer/base_slots/models/savi.py)
    )

    post_recon_loss_w = 1.  # posterior slots image recon (may already include a perceptual blend)
    kld_loss_w = 1e-4  # kld on kernels distribution
