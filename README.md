# LlTRA-Model.

LlTRA stands for: Language to Language Transformer model from the paper "Attention is all you Need", building transformer model: https://github.com/Esmail-ibraheem/Transformer-model, from scratch and use it for translation using pytorch.

### Problem Statement:

In the rapidly evolving landscape of natural language processing (NLP) and machine translation, there exists a persistent challenge in achieving accurate and contextually rich language-to-language transformations. Existing models often struggle with capturing nuanced semantic meanings, context preservation, and maintaining grammatical coherence across different languages. Additionally, the demand for efficient cross-lingual communication and content generation has underscored the need for a versatile language transformer model that can seamlessly navigate the intricacies of diverse linguistic structures.

---

### Goal:

Develop a specialized language-to-language transformer model that accurately translates from the English language to the Arabic language, ensuring semantic fidelity, contextual awareness, cross-lingual adaptability, and the retention of grammar and style. The model should provide efficient training and inference processes to make it practical and accessible for a wide range of applications, ultimately contributing to the advancement of English-to-Arabic language translation capabilities.

---

### Dataset used:

from hugging Face 
https://huggingface.co/datasets/opus_infopankki/viewer/ar-en/train?p=3

---

### Search algorithm used:

Greedy Algorithm for finding which token has the maximum probability.

---

### Training:

I used my drive to upload the project and then connected it to the Google Collab to train it:
#### still training. 
```python
from google.colab import drive

drive.mount('/content/drive')

import os

os.chdir('/content/drive/MyDrive/TrainModel')

%run train.py
```
--- 

check the theoretical part: https://github.com/Esmail-ibraheem/Transformer-model-theoretical-part
