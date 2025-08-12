# Pun2Pun: Benchmarking LLMs on Textual-Visual Chinese-English Pun Translation via Pragmatics Model and Linguistic Reasoning

[![ACL Anthology](https://img.shields.io/badge/ACL%20Anthology-2025.acl--srw.23-1f6feb.svg)](https://aclanthology.org/2025.acl-srw.23/)

## Overview

Pun2Pun is a novel benchmark for evaluating LLMs' capabilities in cross-lingual pun translation/recreation between Chinese and English. The project addresses the complex challenge of preserving both linguistic mechanisms and humorous effects in pun translation through:

- **Constant-Variable Optimization (CVO) Model**: A systematic approach for pun translation strategy
- **Progressive Sub-Tasks**: Four interconnected tasks from classification to translation
- **Comprehensive Evaluation**: Multi-metric assessment including Overlap (Ovl) and Hit metrics
- **Bilingual Dataset**: 5.5k textual puns and 1k visual puns across Chinese and English

## Dataset Structure

### Textual Puns (`data/textual/`)
```
data/textual/
├── en/                     # English puns
│   ├── graphic.json        # Homographic puns
│   ├── phonic.json         # Homophonic puns
│   ├── en_graphic_pun.jsonl      # Task annotations
│   ├── en_phonic_pun.jsonl       
│   └── en_*_explanation.jsonl    
└── zh/                     # Chinese puns
    ├── graphic.json        # Homographic puns
    ├── phonic.json         # Homophonic puns
    ├── zh_graphic_pun.jsonl      # Task annotations
    ├── zh_phonic_pun.jsonl       
    └── zh_*_explanation.jsonl    
```

### Visual Puns (`data/visual/`)
```
data/visual/
├── pun2pun_image/         # Visual pun images
│   ├── en/                # English visual puns
│   └── zh/                # Chinese visual puns
├── en_anno.json           # English annotations
└── zh_anno.json           # Chinese annotations
```

## Dataset Statistics

For textual data, we collected Chinese and English homophonic and homographic puns from multiple sources, with quality assurance and annotations: 

```bibtex
@misc{ChineseHumorSentiment,
  author       = {Liu, Huanyong},
  title        = {ChineseHumorSentiment: Chinese humor sentiment mining including corpus build and NLP methods},
  year         = {2018},
  url          = {https://github.com/liuhuanyong/ChineseHumorSentiment},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/liuhuanyong/ChineseHumorSentiment}}
}
@inproceedings{joke_master,
    title = "Are {U} a Joke Master? Pun Generation via Multi-Stage Curriculum Learning towards a Humor {LLM}",
    author = "Chen, Yang  and
      Yang, Chong  and
      Hu, Tu  and
      Chen, Xinhao  and
      Lan, Man  and
      Cai, Li  and
      Zhuang, Xinlin  and
      Lin, Xuan  and
      Lu, Xin  and
      Zhou, Aimin",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.51",
    pages = "878--890",
}
@inproceedings{simpson2019predicting,
   author    = {Edwin Simpson and Do Dinh, Erik-L{\^{a}}n and
                Tristan Miller and Iryna Gurevych},
   title     = {Predicting Humorousness and Metaphor Novelty with
                {Gaussian} Process Preference Learning},
   booktitle = {Proceedings of the 57th Annual Meeting of the
                Association for Computational Linguistics (ACL 2019)},
   month     = jul,
   year      = {2019},
   pages     = {5716--5728},
}
```

For visual data, since no relevant datasets exist, we manually curated a diverse collection of examples from both Chinese and English public social media sources, consisting of images paired with pun-based captions embedded in them.

| Category | Language | Modality | Homophonic | Homographic |
|----------|----------|----------|------------|-------------|
| Textual | Chinese | Text | 1,154 | 1,490 |
| Textual | English | Text | 1,197 | 1,661 |
| Visual | Chinese | Image | 426 | 74 |
| Visual | English | Image | 155 | 349 |

## Tasks and Terminology

The benchmark consists of four progressive sub-tasks with different naming conventions for textual and visual puns:

| Task | Textual Puns | Visual Puns | Description |
|------|-------------|-------------|-------------|
| **Task I** | Classification (`tag`) | Classification (`tag`) | Identify pun type (homophonic/homographic) |
| **Task II** | Locating (`pun`) | Decomposition (`task2`) | Extract pun elements |
| **Task III** | Decomposition (`explanation`) | Appreciation (`task3` + `task4`) | Analyze pun mechanism |
| **Task IV** | Translation | Translation | Recreate target language pun |

## Usage

### Inference


```bash
# Standard
python infer/{task}/openai_infer.py

# Chain-of-Thought
python infer/{task}/openai_infer_cot.py

# CVO
python infer/{task}/openai_infer_cvo.py
```


### Evaluation

Evaluate model outputs using various metrics:

```bash
# Accuracy
python eval/aacc_pun.py
python eval/aacc_explanation.py

# Semantic similarity
python eval/cosine.py

# Translation quality
python eval/hit.py      # Binary pun detection
python eval/ovl.py      # Overlap metric
```




## Citation

If you use Pun2Pun in your research, please cite:

```bibtex
@article{ma2025pun2pun,
  title={Pun2Pun: Benchmarking LLMs on Textual-Visual Chinese-English Pun Translation via Pragmatics Model and Linguistic Reasoning},
  author={Ma, Yiran Rex and Huang, Shan and Xu, Yuting and Zhou, Ziyu and Wei, Yuanxi},
  journal={WE ARE WORKING ON THIS!},
  year={2025}
}
```
