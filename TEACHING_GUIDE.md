# Instructor's CNN teaching guide — 4 beginner sessions

## Session 1 — How images become neural network inputs

Recall an FFN: features go into neurons; it learns weights using forward pass,
loss, backpropagation and an optimizer. A CNN uses the **same learning process**,
but learns local spatial features in images with filters.

1. An RGB image is a grid: width × height × 3 color channels.
2. Every pixel/channel is usually an integer in 0...255.
3. A **filter/kernel** such as 3×3 scans the image; one learned filter makes a
   feature map that activates for a particular local visual pattern.
4. Multiple filters can learn edges, textures, spots and higher-level patterns.
5. **ReLU** maps negative activations to 0; adds non-linearity.
6. **MaxPool2D** reduces spatial size by taking maxima, saving computation and
   retaining salient activations. Pooling does not learn weights.
7. **Flatten** converts the final 3D feature-map stack into a 1D feature vector.
8. **Dense** combines learned features into class-related information.
9. **Dropout(0.3)** randomly drops about 30% of dense inputs during *training*,
   reducing reliance on particular neurons (may reduce overfitting). It is
   disabled during inference. Not using it is not guaranteed to overfit.
10. **Softmax** generates three scores that sum to approximately 1; `argmax`
    selects the highest score. Scores are not medical/agricultural certainty.

Key distinction: CNN layers learn image patterns, while the earlier FFN still
appears at the end of this classification model as Dense layers.

Model summary (128×128 RGB):

```text
Input             128×128×3
Rescaling          128×128×3       divide by 255
RandomFlip         128×128×3       training only
RandomRotation     128×128×3       training only
Conv2D (16, 3×3)   128×128×16      learn 16 filters
MaxPooling2D        64×64×16       downsample
Conv2D (32, 3×3)    64×64×32       learn 32 filters
MaxPooling2D        32×32×32
Conv2D (64, 3×3)    32×32×64       learn 64 filters
MaxPooling2D        16×16×64
Flatten             16,384
Dense(64)              64
Dropout(0.3)           64          training only
Dense(3, softmax)       3          three classes
```

For 3-class single-label classification, use:
`SparseCategoricalCrossentropy` with integer labels 0,1,2.

## Session 2 — Data and training notebook

1. Download only original color images, separated into three class folders.
2. Verify and inspect corrupt images + class counts.
3. Before creating augmented images, split originals train 70%, val 15%, test 15%.
4. RGB conversion is needed; **BGR→RGB only if using OpenCV**. PIL and Keras
   already load RGB.
5. Resize to 128×128, because model inputs must have uniform dimensions.
6. Put rescaling in the **model** (`1/255`). Do NOT rescale again in API.
7. Augment training images dynamically using modest random flips/rotations;
   never augment validation or test data.
8. Show the model summary, train for a few epochs, compare train/val graphs.
9. Since Healthy class is less common, explain `class_weight` in fit as a
   simple way to increase the loss contribution of underrepresented images.
10. Evaluate once on held-out test images; inspect confusion matrix to see
    which types of leaves are confused.
11. Save `.keras` model and exact ordered class labels in JSON.

## Session 3 — Practical inference + web app

1. Give a truly unseen phone photo, recognize domain shift may reduce quality.
2. Local CLI predicts directly from the `.keras` model.
3. FastAPI exposes `POST /predict` accepting image file; validates size/type.
4. Streamlit UI sends image to FastAPI and renders class/score/advice.
5. Only after this works, show Gemini (optional). CNN classifies; LLM explains.

## Session 4 — Repository and recap before NLP

1. Explain `.venv`, `requirements.txt`, `.gitignore`, `.env.example`, README.
2. Demonstrate clone-first versus init-first GitHub workflows.
3. Commit code only (no secret key or multi-gigabyte data/model files).
4. Recap: CNN learns visual feature maps; NLP later learns text token
   representations and eventually transformer attention.

## Questions to ask the class

- Why is an image represented with three channels?
- What do multiple Conv2D filters learn?
- How does MaxPooling differ from Flatten?
- Why is ReLU needed if we already have convolution?
- What happens if we do not use Dropout?
- Why do we split before augmentation?
- Does augmentation guarantee balanced classes? (No.)
- Why must inference preprocessing match training preprocessing?
- What does a confusion matrix tell us beyond accuracy?
- Why is Gemini not asked to decide the predicted disease?
