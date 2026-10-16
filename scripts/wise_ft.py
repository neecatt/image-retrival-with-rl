#!/usr/bin/env python3
"""WiSE-FT (Wortsman et al., CVPR 2022): interpolate pretrained and fine-tuned weights,
theta(alpha) = (1 - alpha) theta_0 + alpha theta_ft, and choose alpha on held-out validation
(mean of GLDv2-clean val and SfM-120k val leave-one-out mAP). The two checkpoints must share
the architecture (e.g. DINOv2 CLS without projection). Saves every evaluated alpha's scores and the
selected interpolated checkpoint. No benchmark numbers are used for the choice.
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import get_transforms                                    # noqa: E402
from data.dataset_sfm120k import load_gldv2c, load_sfm120k                  # noqa: E402
from evaluate import create_model_from_checkpoint                           # noqa: E402
from evaluation.sfm_val import evaluate_sfm_val                             # noqa: E402


def clean(sd):
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero', required=True)
    ap.add_argument('--ft', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out', required=True, help='output .pth for the selected interpolation')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument('--val_size', type=int, default=448)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    dev = torch.device('cuda')
    root = Path(a.data_root)
    tf = get_transforms(a.val_size, 'eval')
    _, gv = load_gldv2c(str(root / 'gldv2c'), val_transform=tf)
    _, sv = load_sfm120k(str(root / 'sfm120k'), val_transform=tf)
    z, f = torch.load(a.zero, map_location='cpu', weights_only=False), torch.load(a.ft, map_location='cpu', weights_only=False)
    s0, s1 = clean(z['model_state_dict']), clean(f['model_state_dict'])
    assert s0.keys() == s1.keys(), "architectures differ"
    model = create_model_from_checkpoint(a.ft, dev)
    target = getattr(model, '_orig_mod', model)
    res, best = [], None
    for al in a.alphas:
        target.load_state_dict({k: ((1 - al) * s0[k].float() + al * s1[k].float()).to(s1[k].dtype) if s1[k].is_floating_point()
                                else s1[k] for k in s1})
        g = evaluate_sfm_val(model, gv, dev, num_workers=a.workers)['sfm_val_mAP']
        s = evaluate_sfm_val(model, sv, dev, num_workers=a.workers)['sfm_val_mAP']
        res.append({'alpha': al, 'gldv2c_val': g, 'sfm_val': s, 'mean': (g + s) / 2})
        print(json.dumps(res[-1]), flush=True)
        if best is None or res[-1]['mean'] > best['mean']:
            best = res[-1]
            state = dict(f); state['model_state_dict'] = {k: v.cpu() for k, v in target.state_dict().items()}
            state['wise_ft'] = {'alpha': al, 'zero': a.zero, 'ft': a.ft}
            state.pop('optimizer_state_dict', None)
            torch.save(state, a.out)
    Path(a.out).with_suffix('.json').write_text(json.dumps({'results': res, 'selected': best}, indent=1))
    print('selected', best)


if __name__ == '__main__':
    main()
