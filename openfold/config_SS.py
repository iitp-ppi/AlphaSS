import copy
import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def model_config(name, train=False, low_prec=False):
    c = copy.deepcopy(config)
    if name == "model_1":
        pass
    elif name == "model_2":
        pass
    elif name == "model_3":
        c.model.template.enabled = False
    elif name == "model_4":
        c.model.template.enabled = False
    elif name == "model_5":
        c.model.template.enabled = False
    elif name == "model_1_ptm":
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_2_ptm":
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_3_ptm":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_4_ptm":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
    elif name == "model_5_ptm":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        c.loss.tm.weight = 0.1
        c.training.use_pretrained_parameter = True
        c.model.SS_enable = False
        c.loss.SS.weight = 0.0
        
              
   #psh modifying... for adding disulfide bond model..
    elif name == "AlphaSS_ft_model":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        
        c.loss.tm.weight = 0.1
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
        
        c.training.use_pretrained_parameter = True

        c.training.max_epochs = 5
        
        c.training.batch_size = 64
        c.training.learning_rate = 5e-4
        c.model.SS_enable = True


    elif name == "AlphaSS_ft_model_SSloss1":
        c.model.template.enabled = False
        c.model.heads.tm.enabled = True
        
        c.loss.tm.weight = 0.1
        c.loss.violation.weight = 1.
        c.loss.experimentally_resolved.weight = 0.01
        c.loss.SS.weight = 1.0
        
        c.training.use_pretrained_parameter = True

        c.training.max_epochs = 5
        
        c.training.batch_size = 64
        c.training.learning_rate = 5e-4
        c.model.SS_enable = True
        

        
    else:
        raise ValueError("Invalid model name")

    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.chunk_size = None

    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)

    return c


c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
embed_template_torsion_angles = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

config = mlc.ConfigDict(
    {
        # PSH modifying.. for selecting whether or not using pretrained_parameter of AlphaFold2
        "training" : {"use_pretrained_parameter" : True,
                      "batch_size" : 512,
                      "learning_rate" : 5 * 1e-4,
                      "tensorboard_logdir" : '/raid/bis/sehoon/data/tensorboard_logs/editLoss',
                      "freeze_before_structure_module" : False,
                      "freeze_before_evoformer" : False,
                      "freeze_except_SSembedder" : False,
                      "max_epochs" : 5,},
        
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "alt_chi_angles": [NUM_RES, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "extra_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
                    "is_distillation": [],
                    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
                    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
                    "msa_row_mask": [NUM_MSA_SEQ],
                    "no_recycling_iters": [],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "resolution": [],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "template_aatype": [NUM_TEMPLATES, NUM_RES],
                    "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
                    "template_all_atom_positions": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_alt_torsion_angles_sin_cos": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_backbone_rigid_mask": [NUM_TEMPLATES, NUM_RES],
                    "template_backbone_rigid_tensor": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_mask": [NUM_TEMPLATES],
                    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
                    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
                    "template_sum_probs": [NUM_TEMPLATES, None],
                    "template_torsion_angles_mask": [
                        NUM_TEMPLATES, NUM_RES, None,
                    ],
                    "template_torsion_angles_sin_cos": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "true_msa": [NUM_MSA_SEQ, NUM_RES],
                    "use_clamped_fape": [],
                    # PSH modyfying...
                    "disulf_disto" : [
                        NUM_RES, NUM_RES, None,
                    ],
                    "disulf_dist" : [
                        NUM_RES, NUM_RES, None,
                    ],
                },
                "masked_msa": {
                    "profile_prob": 0.1,
                    "same_prob": 0.1,
                    "uniform_prob": 0.1,
                },
                "max_extra_msa": 1024,
                "max_recycling_iters": 3,
                "msa_cluster_features": True,
                "reduce_msa_clusters_by_max_templates": False,
                "resample_msa_in_recycling": True,
                "template_features": [
                    "template_all_atom_positions",
                    "template_sum_probs",
                    "template_aatype",
                    "template_all_atom_mask",
                ],
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "msa",
                    "num_alignments",
                    "seq_length",
                    "between_segment_residues",
                    "deletion_matrix",
                    "no_recycling_iters",
                    # PSH modifying for adding disulf_disto feature to feature
                    "disulf_disto",
                    "disulf_dist",
                ],
                "use_templates": templates_enabled,
                "use_template_torsion_angles": embed_template_torsion_angles,
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_positions",
                    "resolution",
                    "use_clamped_fape",
                    "is_distillation",
                ],
            },
            "predict": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "supervised": False,
                "subsample_recycling": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "subsample_recycling": False,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "subsample_templates": True,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_template_hits": 4,
                "max_templates": 4,
                "shuffle_top_k_prefiltered": 20,
                "crop": True,
                "crop_size": 384,
                "supervised": True,
                "clamp_prob": 0.9,
                "subsample_recycling": True,
                "max_distillation_msa_clusters": 1000,
                "uniform_recycling": True,
            },
            "data_module": {
                "use_small_bfd": False,
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 8,
                },
            },
        },
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
        },
        "model": {
            "name" : 'vanila_openfold',
            "_mask_trans": False,
            "SS_enable" : False,
            "input_embedder": {
                "tf_dim": 22,
                "msa_dim": 49,
                "c_z": c_z,
                "c_m": c_m,
                "relpos_k": 32,
            },
            
            # PSH modifying for taking disulf_Embedder
            "SS_embedder":{
                "c_z" : c_z,
                "c_m" : 1,
            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_m,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
            },
            "template": {
                "distogram": {
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bins": 39,
                },
                "template_angle_embedder": {
                    # DISCREPANCY: c_in is supposed to be 51.
                    "c_in": 57,
                    "c_out": c_m,
                },
                "template_pair_embedder": {
                    "c_in": 88,
                    "c_out": c_t,
                },
                "template_pair_stack": {
                    "c_t": c_t,
                    # DISCREPANCY: c_hidden_tri_att here is given in the supplement
                    # as 64. In the code, it's 16.
                    "c_hidden_tri_att": 16,
                    "c_hidden_tri_mul": 64,
                    "no_blocks": 2,
                    "no_heads": 4,
                    "pair_transition_n": 2,
                    "dropout_rate": 0.25,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "inf": 1e9,
                },
                "template_pointwise_attention": {
                    "c_t": c_t,
                    "c_z": c_z,
                    # DISCREPANCY: c_hidden here is given in the supplement as 64.
                    # It's actually 16.
                    "c_hidden": 16,
                    "no_heads": 4,
                    "inf": 1e5,  # 1e9,
                },
                "inf": 1e5,  # 1e9,
                "eps": eps,  # 1e-6,
                "enabled": templates_enabled,
                "embed_angles": embed_template_torsion_angles,
            },
            "extra_msa": {
                "extra_msa_embedder": {
                    "c_in": 25,
                    "c_out": c_e,
                },
                "extra_msa_stack": {
                    "c_m": c_e,
                    "c_z": c_z,
                    "c_hidden_msa_att": 8,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": 128,
                    "c_hidden_pair_att": 32,
                    "no_heads_msa": 8,
                    "no_heads_pair": 4,
                    "no_blocks": 4,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "clear_cache_between_blocks": True,
                    "inf": 1e9,
                    "eps": eps,  # 1e-10,
                },
                "enabled": True,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
            },    
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 7,
                "trans_scale_factor": 10,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                },
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                "tm": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                    "enabled": False,
                },
                "masked_msa": {
                    "c_m": c_m,
                    "c_out": 23,
                },
                "experimentally_resolved": {
                    "c_s": c_s,
                    "c_out": 37,
                },
            },
        },
        "relax": {
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
        },
        "loss": {
            "SS": {
                "eps": 1e-6,  # 1e-6,
                "weight": 1.0,
            },
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": 0.3,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "weight": 0.01,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "lddt": {
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": 0.01,
            },
            "masked_msa": {
                "eps": eps,  # 1e-8,
                "weight": 2.0,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": eps,  # 1e-8,
                "weight": 0.0,
            },
            "eps": eps,
        },
        "ema": {"decay": 0.999},
    }
)
