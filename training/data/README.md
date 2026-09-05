# Datasets

Nothing in here is committed — the image globs in the repo `.gitignore` exclude
it. This file documents what to put where; `sigtrain setup` checks the layout
and tells you what is missing.

```
data/
├── raw/
│   ├── sign_data/          ← Kaggle dataset, extracted
│   │   ├── train/
│   │   │   ├── 001/        genuine signatures for person 001
│   │   │   ├── 001_forg/   forgeries — skipped by every stage
│   │   │   ├── 002/
│   │   │   └── ...
│   │   └── test/           different people; never seen during training
│   └── stamps/             stamp/seal images used as CycleGAN noise
└── processed/              generated; safe to delete and rebuild
    ├── cyclegan/           trainA/ trainB/ testA/ testB/
    └── verification/       genuine-only person folders
```

## Source

**Signature Verification Dataset** — <https://www.kaggle.com/datasets/robinreni/signature-verification-dataset>
(Robin Reni). Handwritten genuine and forged signatures, PNG, white background.

Alternative with the same layout:
<https://www.kaggle.com/datasets/mallapraveen/signature-matching>

```bash
# with the Kaggle CLI configured
kaggle datasets download -d robinreni/signature-verification-dataset
unzip signature-verification-dataset.zip -d data/raw/
# ensure the result is data/raw/sign_data/{train,test}/
```

## Forgeries

`_forg` folders are excluded everywhere, on purpose. The extractors are trained
as person classifiers over *genuine* signatures and the embedding is reused for
verification; feeding forgeries in as if they were that person's own signature
would teach the model to place them nearby, which is the opposite of useful.

They stay in the raw download because they are the natural test set for a
forgery-detection model, if one is ever added.

## Stamps

Not redistributable, so not scripted. Collect a few dozen images of round
official seals — `data/raw/stamps/*.{png,jpg}` — with white or near-white
backgrounds; `StampAugmentor` removes the background, tints greyscale scans to
the standard red ink, scales them to a DPI-realistic diameter and composites
them with a multiply blend.

Without them the stage warns and continues, and the denoiser simply never
learns to remove seals.

## Why no CSV manifests

Earlier revisions carried five pair-list CSVs (~7.3 MB, ~170k rows). No code
ever read them, each family was internally redundant, and the cyclegan
"train/test" pair lists referenced the same person folders on both sides — so
using them as a split would have leaked identities. The stages walk directories
instead.
