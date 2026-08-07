<div align="center">

[![Project Icon](images/Icon.png)]

# LLM-as-a-Judge Evaluation Harness

**A pipeline wich proporse is to evaluate the decisions made by an LLM chosing the best response for a conversation between another LLM and a human.**


</div>
---

### Motivation 

Searching about the diferents datasets in HuggingFace, I faced the HH-RLHF. This dataset could be analised by an AI to try to chose the best answer, wich was **already been chosen by an human before**.


### Where and why the judge make mistakes?

Based on this question the project was structured.
In search for the answer I used the **HHH dimension multi-label** categorie, to show all parameters **where the judge failed**.

---

### Dataset

I choose the HH-RLHF dataset because the easeniss to work with, and the characteristic he has not been throught an AI model.
For this experiment I used an sample of 300 conversations extracted from the HuggingFace library.
Another similar dataset is the HH-Golden wich was considerated to do this experiment, check the diferences below: 

| | **RLHF version** | **Golden Version** |
|---|---|---|
| **Contaminated by a LLM** | **No** | Yes |
| **Origin** | **Coleted on real conversations** | Based on the RLHF version |
| **Data Quality** |  Contain ambiguities and noises | **Cleaner and aprimorated version created by GPT-4** |
| **Gap** | Small gap between the responses | **Bigger and clearer gap between the responses** |

---

### Methodology (the judge prompt design, categories, statistic used)
#### How the judge prompt was designed
#### Wich HHH I adapted and why
#### Wich statistic is used and why

---

### Reproducibility (Wich model was used, temperature, seed, sample dimension)(the exact parameters)

---

### Results (real Kappa table comparated with the references table, wich was my Kappa, what that means in Land & Koch scale, and **the HHH error table**)

---

### Limitations (sample dimension, LLM judge bias)

---

### How to run (setup, commands)

---

### References (dataset used, framework HHH, papers from Kappa)
