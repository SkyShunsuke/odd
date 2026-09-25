"""Cache full-resolution (256x256) test defect masks for pixel metrics (one dataset pass)."""
import argparse, os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from src.datasets import build_dataset
from cache_latents import DATA_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ds = build_dataset(train=False, **DATA_PARAMS)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                         num_workers=8, pin_memory=False)
    masks, labels, cls, idxs = [], [], [], []
    for batch in tqdm(loader):
        m = batch['mask'].float()
        if m.ndim == 3:
            m = m.unsqueeze(1)
        m = F.interpolate(m, size=(256, 256), mode='nearest').squeeze(1)
        masks.append((m > 0).to(torch.uint8))
        labels.append(batch['label'])
        cls.append(batch['clslabel'])
        idxs.append(batch['index'])
    torch.save(dict(mask256=torch.cat(masks), label=torch.cat(labels),
                    clslabel=torch.cat(cls), index=torch.cat(idxs)),
               os.path.join(args.out, 'test_masks256.pt'))
    print('saved', os.path.join(args.out, 'test_masks256.pt'))


if __name__ == '__main__':
    main()
