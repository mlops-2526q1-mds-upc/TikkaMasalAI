---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for Food-101

<!-- Provide a quick summary of the dataset. -->

The dataset, provided by ethz via Huggingface, contains 101000 food images divided into 101 categories. Each class contains 250 test images as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.


## Dataset Details

### Dataset Description


<!-- Provide a longer summary of what this dataset is. -->

- **Curated by:** ethz (Huggingface)
- **Language(s) (NLP):** English
- **License:** The Food-101 data set consists of images from Foodspotting [1] which are not property of the Federal Institute of Technology Zurich (ETHZ). Any use beyond scientific fair use must be negociated with the respective picture owners according to the Foodspotting terms of use [2].

### Dataset Sources 
- **Repository:** https://huggingface.co/datasets/ethz/food101

## Uses
The intended use of the dataset is to create a machine learning classifier that can detect food dishes given their image. 
Furthermore, we aim to further develop the system by adding other models that can provide descriptions of the dishes, nutritional value or recommend recipes. 
This dataset will serve as a cornerstone to our machine learning system, but could be supplemented with other datasets. 

### Direct Use

- Training an already existing deep-learning model that classifies images with their dish name.


## Dataset Structure

The dataset consists of:
- colored images stored as PIL with max side length of 512 pixels
- labels - 101 unique values stored as int64
<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

### Data Instances

A sample from the training set is provided below:

```python
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x276021C5EB8>,
  'label': 23}
```


### Data Fields
The data instances have the following fields:

- image: A PIL.Image.Image object containing the image. Note that when accessing the image column: dataset[0]["image"] the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the "image" column, i.e. dataset[0]["image"] should always be preferred over dataset["image"][0].
- label: an int classification label.

### Class Label Mapping

<details>
  <summary>Mapping of food classes to labels</summary>

{
  "apple_pie": 0,
  "baby_back_ribs": 1,
  "baklava": 2,
  "beef_carpaccio": 3,
  "beef_tartare": 4,
  "beet_salad": 5,
  "beignets": 6,
  "bibimbap": 7,
  "bread_pudding": 8,
  "breakfast_burrito": 9,
  "bruschetta": 10,
  "caesar_salad": 11,
  "cannoli": 12,
  "caprese_salad": 13,
  "carrot_cake": 14,
  "ceviche": 15,
  "cheesecake": 16,
  "cheese_plate": 17,
  "chicken_curry": 18,
  "chicken_quesadilla": 19,
  "chicken_wings": 20,
  "chocolate_cake": 21,
  "chocolate_mousse": 22,
  "churros": 23,
  "clam_chowder": 24,
  "club_sandwich": 25,
  "crab_cakes": 26,
  "creme_brulee": 27,
  "croque_madame": 28,
  "cup_cakes": 29,
  "deviled_eggs": 30,
  "donuts": 31,
  "dumplings": 32,
  "edamame": 33,
  "eggs_benedict": 34,
  "escargots": 35,
  "falafel": 36,
  "filet_mignon": 37,
  "fish_and_chips": 38,
  "foie_gras": 39,
  "french_fries": 40,
  "french_onion_soup": 41,
  "french_toast": 42,
  "fried_calamari": 43,
  "fried_rice": 44,
  "frozen_yogurt": 45,
  "garlic_bread": 46,
  "gnocchi": 47,
  "greek_salad": 48,
  "grilled_cheese_sandwich": 49,
  "grilled_salmon": 50,
  "guacamole": 51,
  "gyoza": 52,
  "hamburger": 53,
  "hot_and_sour_soup": 54,
  "hot_dog": 55,
  "huevos_rancheros": 56,
  "hummus": 57,
  "ice_cream": 58,
  "lasagna": 59,
  "lobster_bisque": 60,
  "lobster_roll_sandwich": 61,
  "macaroni_and_cheese": 62,
  "macarons": 63,
  "miso_soup": 64,
  "mussels": 65,
  "nachos": 66,
  "omelette": 67,
  "onion_rings": 68,
  "oysters": 69,
  "pad_thai": 70,
  "paella": 71,
  "pancakes": 72,
  "panna_cotta": 73,
  "peking_duck": 74,
  "pho": 75,
  "pizza": 76,
  "pork_chop": 77,
  "poutine": 78,
  "prime_rib": 79,
  "pulled_pork_sandwich": 80,
  "ramen": 81,
  "ravioli": 82,
  "red_velvet_cake": 83,
  "risotto": 84,
  "samosa": 85,
  "sashimi": 86,
  "scallops": 87,
  "seaweed_salad": 88,
  "shrimp_and_grits": 89,
  "spaghetti_bolognese": 90,
  "spaghetti_carbonara": 91,
  "spring_rolls": 92,
  "steak": 93,
  "strawberry_shortcake": 94,
  "sushi": 95,
  "tacos": 96,
  "takoyaki": 97,
  "tiramisu": 98,
  "tuna_tartare": 99,
  "waffles": 100
}

</details>


## Citation 

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```python
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```

