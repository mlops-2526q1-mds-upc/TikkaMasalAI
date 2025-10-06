---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for vgg16

<!-- Provide a quick summary of what the model is/does. -->

VGG16 is a convolutional neural network architecture that is designed for image classification tasks. It is known for its deep architecture and use of small convolutional filters.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** pytorch 
- **Model type:** Image classification model
- **Language(s) (NLP):** Python

### Model Sources

<!-- Provide the basic links for the model. -->
- **Paper:** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) by Karen Simonyan and Andrew Zisserman.

## Uses

This model is intended for image classification tasks, such as:

- Assisting image recognition in mobile apps for various domains.
- Supporting automated tagging and organization of images in datasets.
- Enabling menu digitization and visual search.
- Facilitating research in computer vision applications.
- Providing educational tools for image identification.

Foreseeable users include developers building image-related applications, researchers in computer vision, and organizations managing large image datasets.

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be directly used to classify images into various categories without additional fine-tuning. Users can input an image and receive a predicted label, making it suitable for integration into applications such as mobile image recognition, automated dataset labeling, and menu digitization tools.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is not intended for:
- Medical diagnosis or health-related decision making.
- Identifying allergens or dietary restrictions.
- Use in safety-critical systems.
- Classification of non-image data or images with multiple objects.
- Detecting image quality, spoilage, or contamination.



## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

More information needed

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the following code snippet to load and use the model:

```python
from src.models.vgg16 import VGG16
from PIL import Image

# Load model and processor from local folder
vgg16_food_model = VGG16()

# Load and preprocess an image
image = Image.open("path_to_your_image.jpg")
inputs = vgg16_food_model.preprocess(images=image, return_tensors="pt")

# Run inference
outputs = vgg16_food_model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
print(f"Predicted class: {predicted_class}")
```

## Training Details

### Training Data

- **Dataset:** ImageNet ILSVRC-2012 — 1.3M training images across 1000 classes.  
- **Data Augmentation:** Random resized crops (224×224), horizontal flips, and random RGB color jittering.

### Training Procedure

#### Preprocessing

- Images resized isotropically so the **shortest side = S** (fixed at 256 / 384 or randomly sampled from **[256, 512]** for scale jittering).  
- Per-pixel **global mean RGB subtraction** applied.

#### Training Hyperparameters

- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum  
- **Batch Size:** 256  
- **Momentum:** 0.9  
- **Weight Decay (L2):** 5e-4  
- **Dropout:** 0.5 on first two fully-connected layers  
- **Learning Rate Schedule:**  
  - Initial LR = 0.01  
  - Reduced by ×0.1 upon validation plateau  
  - Total of 3 reductions → ~74 epochs / 370k iterations  
- **Weight Initialization:**  
  - Shallow model (A) from Gaussian init (std=0.01)  
  - Deeper models initialized from shallower ones (partial layer transfer)  
- **Training Regime:** fp32

#### Speeds, Sizes, Times

- **Hardware:** 4× NVIDIA Titan Black GPUs  
- **Training Time:** 2–3 weeks per model  
- **Parallelism:** Synchronous data-parallel multi-GPU

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- ImageNet validation set (50k images)

#### Factors

- **Single-scale vs multi-scale** testing  
- **Dense convolutional evaluation vs. multi-crop (50-crop at 3 scales)**

#### Metrics

- **Top-1 Error (%)**  
- **Top-5 Error (%)** — standard ILSVRC benchmarks


### Results

Results taken from the original paper on the VGG16 model:

- Performance at a single test scale

| ConvNet config. (Table 1) | Smallest image side train (S) | Smallest image side test (Q) | Top-1 val. error (%) | Top-5 val. error (%) |
|---------------------------|-------------------------------:|-----------------------------:|---------------------:|---------------------:|
| A                         | 256                            | 256                          | 29.6                 | 10.4                 |
| A-LRN                     | 256                            | 256                          | 29.7                 | 10.5                 |
| B                         | 256                            | 256                          | 28.7                 | 9.9                  |
| C                         | 256                            | 256                          | 28.1                 | 9.4                  |
|                           | 384                            | 384                          | 28.1                 | 9.3                  |
|                           | [256;512]                      | 384                          | 27.3                 | 8.8                  |
| D                         | 256                            | 256                          | 27.0                 | 8.8                  |
|                           | 384                            | 384                          | 26.8                 | 8.7                  |
|                           | [256;512]                      | 384                          | 25.6                 | 8.1                  |
| E                         | 256                            | 256                          | 27.3                 | 9.0                  |
|                           | 384                            | 384                          | 26.9                 | 8.7                  |
|                           | [256;512]                      | 384                          | **25.5**             | **8.0**              |



## Environmental Impact

The original paper does not report energy consumption or carbon emissions. However, based on the described training setup, the following estimates and assumptions can be made:

- **Hardware Type:** 4× NVIDIA Titan Black GPUs (single machine, data-parallel training)
- **Hours Used:** Approximately 2–3 weeks per model (~350–500 GPU-hours per GPU) → ~1,400–2,000 total GPU-hours
- **Cloud Provider:** Not applicable — training was conducted on-premise academic hardware
- **Compute Region:** Not specified

Carbon emissions were not measured, but using a standard emission factor for consumer GPU compute (~0.4–0.6 kg CO₂e per GPU-hour depending on energy mix), total emissions for training a single model could be roughly estimated in the range of **560–1200 kg CO₂e**. This is comparable to **a round-trip transatlantic flight per model**.

These numbers should be considered approximate and illustrative rather than exact. Future replications should include explicit energy tracking for more accurate accounting.

