---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for Food-101-93M

<!-- Provide a quick summary of what the model is/does. -->

Food-101-93M is a fine-tuned image classification model built on top of google/siglip2-base-patch16-224 using the SiglipForImageClassification architecture. It is trained to classify food images into one of 101 popular dishes, derived from the Food-101 dataset.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** prithivMLmods (Huggingface)
- **Model type:** Image classification model
- **Language(s) (NLP):** Python


### Model Sources 

<!-- Provide the basic links for the model. -->

- **Hugging Face:** https://huggingface.co/prithivMLmods/Food-101-93M

## Uses

This model is intended for food image classification tasks, such as:

- Assisting food recognition in mobile apps for calorie tracking or meal logging.
- Supporting automated tagging and organization of food images in datasets.
- Enabling restaurant menu digitization and visual search.
- Facilitating research in food-related computer vision applications.
- Providing educational tools for culinary identification.

Foreseeable users include developers building food-related applications, researchers in computer vision, and organizations managing large food image datasets.

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be directly used to classify food images into one of 101 categories without additional fine-tuning. Users can input a food image and receive a predicted dish label, making it suitable for integration into applications such as mobile food recognition, automated dataset labeling, and menu digitization tools.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is not intended for:

- Medical diagnosis or health-related decision making.
- Identifying food allergens or dietary restrictions.
- Use in safety-critical systems.
- Classification of non-food images or images with multiple food items.
- Detecting food quality, spoilage, or contamination.
- Any application involving personal or sensitive information without proper consent.
- Generating nutritional information or portion sizes from images.

Misuse or application outside these intended scopes may result in inaccurate or misleading outputs.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

More information needed

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import SiglipProcessor, SiglipForImageClassification
from PIL import Image
import requests

# Load model and processor
model_name = "prithivMLmods/Food-101-93M"
processor = SiglipProcessor.from_pretrained(model_name)
model = SiglipForImageClassification.from_pretrained(model_name)

# Load an image
url = "https://example.com/food_image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess and predict
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class_idx = outputs.logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

print(f"Predicted dish: {predicted_label}")
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- Food101 data set: https://huggingface.co/datasets/ethz/food101

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

Information not provided by model creator.



#### Training Hyperparameters

- **Training regime:** Not provided by model creator. <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

- Food101 data set: https://huggingface.co/datasets/ethz/food101

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->
Not provided by model creator.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Not provided by model creator.

### Results

Not provided by model creator.

#### Summary

Not provided by model creator.


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Not provided by model creator. 