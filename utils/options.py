from argparse import ArgumentParser

from runtime_spec import (
    CLIP_INPUT_SIZE,
    EMBEDDING_REFERENCE_SIZE,
    GENERATOR_MLP_LAYERS,
    GENERATOR_OUTPUT_SIZE,
    HAIR_CLASS_INDEX,
    LATENT_LAYER_COUNT,
    LATENT_STYLE_DIM,
    SEGMENTATION_BACKBONE_SIZE,
    SEGMENTATION_CLASSES,
)


class Options:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--stylegan_path", default="pretrained_models/ffhq.pt", type=str)
        self.parser.add_argument("--seg_path", default="pretrained_models/seg.pth", type=str)
        self.parser.add_argument("--bald_path", default="pretrained_models/bald_proxy.pt", type=str)
        self.parser.add_argument("--ffhq_pca_path", default="pretrained_models/ffhq_PCA.npz", type=str)
        self.parser.add_argument("--output_size", default=GENERATOR_OUTPUT_SIZE, type=int)
        self.parser.add_argument("--embedding_size", default=EMBEDDING_REFERENCE_SIZE, type=int)
        self.parser.add_argument("--segmentation_size", default=SEGMENTATION_BACKBONE_SIZE, type=int)
        self.parser.add_argument("--clip_input_size", default=CLIP_INPUT_SIZE, type=int)
        self.parser.add_argument("--segmentation_classes", default=SEGMENTATION_CLASSES, type=int)
        self.parser.add_argument("--hair_class_index", default=HAIR_CLASS_INDEX, type=int)
        self.parser.add_argument("--generator_style_dim", default=LATENT_STYLE_DIM, type=int)
        self.parser.add_argument("--generator_mlp_layers", default=GENERATOR_MLP_LAYERS, type=int)
        self.parser.add_argument("--generator_latent_count", default=LATENT_LAYER_COUNT, type=int)

        self.parser.add_argument("--W_steps", default=1100, type=int)
        self.parser.add_argument("--FS_steps", default=250, type=int)
        self.parser.add_argument("--lr_embedding", default=0.01, type=float)
        self.parser.add_argument("--l2_lambda_embedding", default=1.0, type=float)
        self.parser.add_argument("--percept_lambda_embedding", default=1.0, type=float)
        self.parser.add_argument("--p_norm_lambda_embedding", default=0.001, type=float)

        self.parser.add_argument("--no_aug_clip_loss_text", default=False, action="store_true")
        self.parser.add_argument("--clip_lambda_text", default=1.0, type=float)
        self.parser.add_argument("--hair_mask_lambda_text", default=1.0, type=float)
        self.parser.add_argument("--lr_text", default=0.01, type=float)
        self.parser.add_argument("--steps_text", default=200, type=int)
        self.parser.add_argument("--visual_num_text", default=1, type=int)

    def parse(self, jupyter=False):
        if jupyter:
            return self.parser.parse_args(args=[])
        return self.parser.parse_args()
