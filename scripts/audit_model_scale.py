"""Export every learned Spike-DINO tensor and architecture-scale summary."""
import argparse
import csv
import json
from pathlib import Path

from spike_dino.dinov3 import SpikingDinoV3Prototype


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch-size', type=int, choices=(8, 16), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    model = SpikingDinoV3Prototype(
        image_size=128, patch_size=args.patch_size, dim=384, depth=12,
        heads=6, head_hidden=2048, head_layers=3, bottleneck=256,
        prototypes=65536, prototype_readout='cosine',
        attention_variant='matrix_recode', cls_aggregation='spike_pool',
        residual_policy='layer_scale', norm_policy='layer_norm')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{'name': name, 'shape': '×'.join(map(str, tensor.shape)),
             'elements': tensor.numel(), 'trainable': tensor.requires_grad}
            for name, tensor in model.named_parameters()]
    path = args.output_dir / f'parameter_inventory_patch{args.patch_size}.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    student = model.student
    summary = {
        'image_size': 128, 'sensor_channels': 2, 'time_steps': 4,
        'patch_size': args.patch_size,
        'global_patch_grid': 128 // args.patch_size,
        'global_patch_tokens': (128 // args.patch_size) ** 2,
        'sequence_tokens_including_cls': 1 + (128 // args.patch_size) ** 2,
        'register_tokens': 0,
        'depth': 12, 'embed_dim': 384, 'heads': 6,
        'head_dim': 64, 'mlp_hidden_dim': 1536,
        'patch_embedding_weight_shape': list(
            student.backbone.patch_embed.weight.shape),
        'qkv_weight_shape_each': list(
            student.backbone.blocks[0].attn.q.linear.weight.shape),
        'mlp_fc1_weight_shape': list(
            student.backbone.blocks[0].fc1.linear.weight.shape),
        'mlp_fc2_weight_shape': list(
            student.backbone.blocks[0].fc2.linear.weight.shape),
        'head_hidden_dim': 2048, 'head_bottleneck_dim': 256,
        'prototypes_each_head': 65536,
        'head_prototype_weight_shape': list(
            student.dino_head.prototypes.shape),
        'backbone_parameters': sum(p.numel() for p in
                                   student.backbone.parameters()),
        'dino_head_parameters': sum(p.numel() for p in
                                    student.dino_head.parameters()),
        'ibot_head_parameters': sum(p.numel() for p in
                                    student.ibot_head.parameters()),
        'student_parameters': sum(p.numel() for p in student.parameters()),
        'ema_teacher_parameters': sum(p.numel() for p in model.teacher.parameters()),
        'student_plus_teacher_parameters': sum(p.numel() for p in
                                               model.parameters()),
        'learned_tensor_count_student_and_teacher': len(rows),
        'architecture_notes': [
            'Q/K/V and MLP hidden inputs are binary spikes.',
            'LayerScale residual currents, input histograms, normalizations, and prototype logits are real-valued.',
            'Attention is spike matrix recoding, not ViT softmax attention.',
            'Learned absolute position current is used; no RoPE or register tokens.',
            'Current diagnostic has two global crops and zero local crops.',
        ],
    }
    (args.output_dir / f'module_summary_patch{args.patch_size}.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
