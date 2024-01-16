# LlTRA-Model.
# ![Transformer drawio](https://github.com/Esmail-ibraheem/LlTRA-Model/assets/113830751/564c75bf-d016-4e6f-9de3-b3cd19bfc743)


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
```python
def Get_configuration():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-4,
        "sequence_length": 100,
        "d_model": 512,
        "datasource": 'opus_infopankki',
        "source_language": "ar",
        "target_language": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
```
---

### Search algorithm used:

Greedy Algorithm for finding which token has the maximum probability.

---

### Training:

I used my drive to upload the project and then connected it to the Google Collab to train it:

pre-trained model in collab.

hours of training: 4 hours.

---

### Some Results:

    SOURCE: العائلات الناطقة بلغة أجنبية لديها الحق في خدمات الترجمة عند اللزوم.
    TARGET: A foreign-language family is entitled to interpreting services as necessary.
    PREDICTED: in a native language is provided by the services of the services for the elderly .
    --------------------------------------------------------------------------------
    SOURCE: قمت بارتكاب جرائم وتُعتبر بأنك خطير على النظام أو الأمن العام.
    TARGET: you have committed crimes and are considered a danger to public order or safety
    PREDICTED: you have committed crimes and are considered a danger to public order or safety
    --------------------------------------------------------------------------------
    SOURCE: عندما تلتحق بالدراسة، فستحصل على الحق في إنجاز كلتا الدرجتين العلميتين.
    TARGET: When you are accepted into an institute of higher education, you receive the right to complete both degrees.
    PREDICTED: When you have a of residence , you will receive a higher education degree .
    --------------------------------------------------------------------------------
    SOURCE: اللجنة لا تتداول حالات التهميش والتمييز المتعلقة بالعمل.
    TARGET: The Tribunal does not handle cases of employment-related discrimination.
    PREDICTED: The does not have to pay and the work .
    --------------------------------------------------------------------------------
    SOURCE: يجب عليك أيضاً أن تستطيع إثبات على سبيل المثال بالوصفة الطبية أو بالتقرير الطبي بأن الغرض من الدواء هو استخدامك أنت الشخصي.
    TARGET: In addition, you must be able to prove with a prescription or medical certificate, for example, that the medicine is intended for your personal use.
    PREDICTED: You must also have to prove your identity with a friend or friend , for example , that the medicine is intended for your personal use .
    --------------------------------------------------------------------------------
    SOURCE: إذا كان لديك ترخيص إقامة في فنلندا، ولكن لم تُمنح ترخيص إقامة استمراري، فسوف تصدر دائرة شؤون الهجرة قراراً بالترحيل.
    TARGET: If you already have a residence permit in Finland but are not granted a residence permit extension, the Finnish Immigration Service makes a deportation decision.
    PREDICTED: If you have a residence permit in but are not granted a residence permit , the Service makes a decision .
--- 

check the theoretical part: [Theoretical part](https://github.com/Esmail-ibraheem/Transformer-model-theoretical-part)
developing process: 
1. https://youtu.be/Nm5CUo7ol18?si=5U3m4IM7crJ23Zzu
2. https://youtu.be/O-jfimyP6Tw?si=ucPweGo2b7gh2rrI
