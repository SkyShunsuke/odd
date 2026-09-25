"""Cache efficientnet-b4 latents for MVTec-AD (all 15 classes), train + test.

Backbone is frozen, so latents are checkpoint-independent. Run once from repo root:
  python cache_latents.py --out <dir> [--device cuda:0]
Saves:
  train_latents.pt: {'z': (N,C,H,W) fp16, 'clslabel': (N,), 'path': [str]}
  test_latents.pt:  {'z': ..., 'clslabel': ..., 'label': (N,), 'index': (N,), 'path': [str],
                     'mask16': (N,16,16) uint8 downsampled defect masks}
"""
import argparse, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from src.datasets import build_dataset
from src.backbones import get_backbone
from src.utils.adeval.eval_utils import extract_features

DATA_PARAMS = dict(
    category='all', data_root='data/mvtec_ad', dataset_name='mvtec_ad_all',
    img_size=256, num_workers=8, persistent_workers=False, pin_memory=True,
    test_batch_size=16, train_batch_size=16, transform_type='imagenet',
)
BB_PARAMS = dict(
    model_name='efficientnet-b4', normalization=None,
    outblocks=[1, 5, 9, 21], outstrides=[2, 4, 8, 16], pretrained=True,
)


def dump(ds, fe, device, out_path, is_test, bs=16):
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                         num_workers=8, pin_memory=True)
    zs, cls, labels, idxs, paths, masks = [], [], [], [], [], []
    for batch in tqdm(loader, desc=out_path):
        imgs = batch['img'].to(device, non_blocking=True)
        with torch.no_grad():
            z, _ = extract_features(fe, imgs, device)
        zs.append(z.half().cpu())
        cls.append(batch['clslabel'])
        paths.extend(batch['path'])
        if is_test:
            labels.append(batch['label'])
            idxs.append(batch['index'])
            m = batch['mask'].float()
            if m.ndim == 3:
                m = m.unsqueeze(1)
            m16 = F.interpolate(m, size=(16, 16), mode='area')
            masks.append((m16.squeeze(1) > 0).to(torch.uint8))
    out = dict(z=torch.cat(zs), clslabel=torch.cat(cls), path=paths)
    if is_test:
        out.update(label=torch.cat(labels), index=torch.cat(idxs), mask16=torch.cat(masks))
    torch.save(out, out_path)
    print(f'saved {out_path}: z {out["z"].shape}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    fe = get_backbone(**BB_PARAMS).to(device).eval()
    train_ds = build_dataset(train=True, **DATA_PARAMS)
    test_ds = build_dataset(train=False, **DATA_PARAMS)
    ncls = len(test_ds.datasets)
    print(f'num_classes={ncls}, ntrain={len(train_ds)}, ntest={len(test_ds)}')
    names = test_ds.datasets[0].labels_to_names
    torch.save(names, os.path.join(args.out, 'class_names.pt'))
    dump(train_ds, fe, device, os.path.join(args.out, 'train_latents.pt'), False)
    dump(test_ds, fe, device, os.path.join(args.out, 'test_latents.pt'), True)


if __name__ == '__main__':
    main()
