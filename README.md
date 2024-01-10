# LlTRA-Model.

LlTRA stands for: Language to Language Transformer model from the paper "Attention is all you Need", building transformer model:[Transformer model](https://github.com/Esmail-ibraheem/Transformer-model) from scratch and use it for translation using pytorch.

### Problem Statement:

In the rapidly evolving landscape of natural language processing (NLP) and machine translation, there exists a persistent challenge in achieving accurate and contextually rich language-to-language transformations. Existing models often struggle with capturing nuanced semantic meanings, context preservation, and maintaining grammatical coherence across different languages. Additionally, the demand for efficient cross-lingual communication and content generation has underscored the need for a versatile language transformer model that can seamlessly navigate the intricacies of diverse linguistic structures.

---

### Goal:

Develop a specialized language-to-language transformer model that accurately translates from the Arabic language to the English language, ensuring semantic fidelity, contextual awareness, cross-lingual adaptability, and the retention of grammar and style. The model should provide efficient training and inference processes to make it practical and accessible for a wide range of applications, ultimately contributing to the advancement of Arabic-to-English language translation capabilities.

---

### Dataset used:

from hugging Face 
[huggingface/opus_infopankki](https://huggingface.co/datasets/opus_infopankki/viewer/ar-en/train?p=3)

---

### Configuration:

this is the settings of the model, You can customize the source and target languages, sequence lengths for each, the number of epochs, batch size, and more.

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
hours of training until now: 8 hours.

---

### Results:

    SOURCE: العائلات الناطقة بلغة أجنبية لديها الحق في خدمات الترجمة عند اللزوم.
    TARGET: A foreign-language family is entitled to interpreting services as necessary.
    PREDICTED: in a native language is provided by the services of the services for the elderly .
    --------------------------------------------------------------------------------
    SOURCE: يمكن لمواطني الاتحاد الأوروبي (EU) والمنطقة الاقتصادية الأوروبية (ETA) أن يعلنوا عن أنفسهم كباحثين عن العمل وذلك بواسطة الخدمة الشبكية لمكتب العمل والموارد المعيشية في قسم "Oma asiointi" المعاملات الشخصية.
    TARGET: To find web pages for jobs on the Internet, write “avoimet työpaikat” (vacancies) in the search engine’s text field. Many web pages for jobs allow you to fill in and send a job application and to enclose your CV.
    PREDICTED: of : ( ) and , , and .
    --------------------------------------------------------------------------------
    SOURCE: عندما تلتحق بالدراسة، فستحصل على الحق في إنجاز كلتا الدرجتين العلميتين.
    TARGET: When you are accepted into an institute of higher education, you receive the right to complete both degrees.
    PREDICTED: When you have a of residence , you will receive a higher education degree .
    --------------------------------------------------------------------------------
    SOURCE: عندما استقلت فنلندا سنة 1917، أصبحت هلسنكي العاصمة لجمهورية فنلندا.
    TARGET: When Finland gained its independence in 1917, Helsinki became the capital of the republic.
    PREDICTED: When gained its independence in , the became the capital of .
    --------------------------------------------------------------------------------
    SOURCE: مركز الضمان التقاعدي يقدم النصيحة لك، عندما تطلب التقاعد من الخارج.
    TARGET: The Finnish Centre for Pension will give you advice for applying for pension abroad.
    PREDICTED: The for will apply for a when you apply for .
    --------------------------------------------------------------------------------
    SOURCE: اللجنة لا تتداول حالات التهميش والتمييز المتعلقة بالعمل.
    TARGET: The Tribunal does not handle cases of employment-related discrimination.
    PREDICTED: The does not have to pay and the work .
    --------------------------------------------------------------------------------
    SOURCE: القاسم المشترك لهذه الغرف هو عمل القهوة وإمكانية استعمال الكومبيوتر المجهز باشتراك الإنترنت مجاناً كذلك يمكن في عديد من هذه الغرف التمتع بالوجبات المحضرة في الغرفة بسعر معقول والتي تقدم في أيام العمل.
    TARGET: All centres have a cafe and the opportunity to use a computer with a free-of-charge Internet connection. Most resident centres also offer the opportunity to enjoy an affordable lunch prepared at the centre that is served on weekdays.
    PREDICTED: All centres are open to and at a . Many centres also offer an opportunity to use a at a centre in the centre where are.
    --------------------------------------------------------------------------------
    SOURCE: يجب عليك أيضاً أن تستطيع إثبات على سبيل المثال بالوصفة الطبية أو بالتقرير الطبي بأن الغرض من الدواء هو استخدامك أنت الشخصي.
    TARGET: In addition, you must be able to prove with a prescription or medical certificate, for example, that the medicine is intended for your personal use.
    PREDICTED: You must also have to prove your identity with a friend or friend , for example , that the medicine is intended for your personal use .
--- 

check the theoretical part: [Theoretical part](https://github.com/Esmail-ibraheem/Transformer-model-theoretical-part)

