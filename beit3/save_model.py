from utils import load_and_save_model_for_inference
from timm.models import create_model
import modeling_finetune
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BEiT load model from checkpoint and save', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--checkpoint_activations', action='store_true', default=None, 
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--checkpoint_path', default='output/checkpoint-best/mp_rank_00_model_states.pt', type=str, metavar='MODEL',
                        help='path to checkpoint')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    args = parser.parse_args()
    model = create_model(
        args.model+"_vqav2",
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
        num_classes=args.nb_classes
    )
    load_and_save_model_for_inference(args, model)