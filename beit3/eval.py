from utils import load_and_save_model_for_inference, compute_metrics
from timm.models import create_model
from datasets import create_downstream_dataset
import modeling_finetune
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BEiT load model from checkpoint and save', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str)
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--checkpoint_activations', action='store_true', default=None)
    parser.add_argument('--model_path', default='model.pth', type=str)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--task', type=str, required=True,
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps', 'imagenet'])

    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--sentencepiece_model', type=str, required=True)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str)
    parser.add_argument('--root_folder', default='', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--dist_eval', action='store_true', default=False)

    args = parser.parse_args()
    data_loader_test = create_downstream_dataset(args, is_eval=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    with open("answer2label.json", mode="r", encoding="utf-8") as f:
        label2answer = json.load(f)

    model = create_model(
        args.model+"_vqav2",
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
        num_classes=args.nb_classes
    )

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    if "model" in checkpoint.keys():
        model.load_state_dict(checkpoint["model"])
        checkpoint = checkpoint["model"]
    elif "module" in checkpoint.keys():
        model.load_state_dict(checkpoint["module"])
        checkpoint = checkpoint["module"]
    else:
        model.load_state_dict(checkpoint)

    # Ensure correct loading
    model = model.half().to(device)  # Convert to FP16 for AMP
    model.eval()

    eval_logits = []
    eval_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader_test, desc="Eval ", leave=False):
            for tensor_key in data.keys():
                if tensor_key == "language_tokens":
                    data[tensor_key] = data[tensor_key].to(torch.long).to(device)  # Convert to LongTensor
                else:
                    data[tensor_key] = data[tensor_key].half().to(device)  # Keep FP16 for other inputs
    
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(
                    image=data["image"], 
                    question=data["language_tokens"],  # Now correctly in LongTensor
                    padding_mask=data["padding_mask"]
                )
                batch_size = data["language_tokens"].shape[0]
                eval_logits.extend(logits.cpu().numpy())
                eval_labels.extend(data["labels"].cpu().numpy())


        eval_metrics = compute_metrics(np.array(eval_logits), np.array(eval_labels))
        for key, value in eval_metrics.items():
            print(f"{key}: {value}")
