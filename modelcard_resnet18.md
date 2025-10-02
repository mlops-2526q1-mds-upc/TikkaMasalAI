---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for resnet18
This model card is based on the original paper and repository for ResNet-18.

<!-- Provide a quick summary of what the model is/does. -->

ResNet-18 is a convolutional neural network architecture that is designed for image classification tasks. It is known for its deep residual learning framework, which helps to train very deep networks by using skip connections.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** pytorch 
- **Model type:** Image classification model
- **Language(s) (NLP):** Python

### Model Sources

<!-- Provide the basic links for the model. -->

- **Hugging Face:** https://huggingface.co/pytorch/resnet18 
- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
- **Repository:** https://github.com/KaimingHe/deep-residual-networks?tab=readme-ov-file

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
from src.models.resnet18 import ResNet18
from PIL import Image

# Load model and processor from local folder
resnet18_food_model = ResNet18()

# Load and preprocess an image
image = Image.open("path_to_your_image.jpg")
inputs = resnet18_food_model.preprocess(images=image, return_tensors="pt")

# Run inference
outputs = resnet18_food_model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
print(f"Predicted class: {predicted_class}")
```

## Training Details

### Training Data

- **ImageNet 2012 classification dataset** - 1.28 million training images across 1000 classes
- Images resized with shorter side randomly sampled in [256, 480] for scale augmentation
- 224×224 crops randomly sampled from images or horizontal flips
- Per-pixel mean subtraction and standard color augmentation applied

### Training Procedure

#### Training Hyperparameters

- **Training regime:** SGD with backpropagation
- **Batch size:** 256
- **Initial learning rate:** 0.1
- **Learning rate schedule:** Divided by 10 when error plateaus
- **Weight decay:** 0.0001
- **Momentum:** 0.9
- **Iterations:** Up to 600,000
- **Batch normalization:** Applied after each convolution and before activation
- **Dropout:** Not used
- **Weight initialization:** As described in He et al. (2015)

#### Speeds, Sizes, Times

- **FLOPs:** 1.8×10⁹ (1.8 billion multiply-add operations)
- **Depth:** 18 weighted layers
- **Parameters:** Exact count not specified in paper, but relatively compact compared to deeper variants

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **ImageNet 2012 validation set** - 50,000 images
- **ImageNet 2012 test set** - 100,000 images (via test server)
- **Standard 10-crop testing** applied for comparison studies
- For best results: fully-convolutional form with multi-scale averaging (shorter side in {224, 256, 384, 480, 640})

#### Factors

- Model depth (comparing 18-layer vs 34-layer architectures)
- Network type (plain vs residual)
- Training and validation error progression

#### Metrics

- **Top-1 error rate:** Primary classification accuracy metric
- **Top-5 error rate:** Standard ImageNet evaluation metric
- **Training error:** Used to analyze optimization behavior and degradation problems
- **Validation error:** Used to measure generalization performance

**Note:** According to Table 2 in the paper, ResNet-18 achieved:
- **Top-1 error:** 27.88% on ImageNet validation (10-crop testing)
- Comparable accuracy to plain 18-layer network but with faster convergence

### Results

From the repo used for the original paper on deep residual networks: https://github.com/KaimingHe/deep-residual-networks?tab=readme-ov-file. We haven't found any information on evaluation results provided by the model creator for specifically this resnet18 model.

- Curves on ImageNet (solid lines: 1-crop val error; dashed lines: training error):

![alt text](image.png)

- 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

| Model      | Top-1 Error | Top-5 Error |
|------------|-------------|-------------|
| VGG-16     | 28.5%       | 9.9%        |
| ResNet-50  | 24.7%       | 7.8%        |
| ResNet-101 | 23.6%       | 7.1%        |
| ResNet-152 | 23.0%       | 6.7%        |


- 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

| Model      | Top-1 Error | Top-5 Error |
|------------|-------------|-------------|
| ResNet-50  | 22.9%       | 6.7%        |
| ResNet-101 | 21.8%       | 6.1%        |
| ResNet-152 | 21.4%       | 5.7%        |



## Environmental Impact

The original paper does not report energy consumption or carbon emissions. However, based on the described training setup, the following estimates and assumptions can be made:

- **Hardware Type:** [GPUs - specific model not specified in paper]
- **Hours used:** Approximately 60 hours (estimated from 600,000 iterations with batch size 256 on ImageNet)
- **Cloud Provider:** [Not specified - research was conducted at Microsoft Research]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

**Training Considerations:**
- ResNet-18 is relatively efficient compared to deeper variants (152-layer), requiring only 1.8 billion FLOPs
- The model was trained from scratch without pre-trained weights
- Batch normalization enabled faster convergence, potentially reducing training time
- No mixed precision training was used (standard FP32)
- The 18-layer architecture represents a more computationally efficient alternative to deeper networks while maintaining competitive accuracy


