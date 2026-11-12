# Dataset download and setup

**Dataset source (Kaggle):**
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

**Choose the ORIGINAL color version, not augmented or segmented images.**
The Kaggle download may contain folders named `color`, `grayscale`, and `segmented`.
Locate the `color` folder anywhere inside the extracted download.

From `color`, copy ONLY these THREE original folders into your project:

```text
data/raw/
├── Potato___Early_blight/  (images)
├── Potato___Late_blight/   (images)
└── Potato___healthy/       (images)
```

Image names inside folders can be anything. There is no CSV needed: the
folder name is the label. Some versions may use different folder names;
rename to the exact spellings above **before** preparing the data.

## Why not use already augmented images?
Augmented copies of the same original leaf might enter both train and test,
making test accuracy appear better than it truly is.

## Create an original-image split

```bash
python prepare_data.py
```

The script validates images, skips exact-duplicate files, converts to RGB,
and makes reproducible approximate 70% train / 15% val / 15% test folders.
It does not change `data/raw`. If you change the raw data and intentionally
want a new split:

```bash
python prepare_data.py --force
```

`--force` deletes **only generated** files under `data/split`.

## Unseen photo
Put your own potato leaf JPG under `data/unseen/leaf.jpg`. Do not copy an
image from training, validation, or test: then it would not be truly unseen.
Remember that a phone image may perform worse than controlled dataset images.
