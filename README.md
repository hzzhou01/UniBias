# UniBias
This repository contains the code for our [NeurIPS 2024 paper:](https://arxiv.org/abs/2405.20612)

**"UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation" (NeurIPS 2024)**

---
## Dependencies

We use python 3.8 and pytorch 2.0.1. You can use ```pip install -r requirements.txt``` to install the required libraries.

## Data

Running the code will automatically download datasets from Huggingface.

## Run Experiment

To run UniBias using the following command:

```bash
python main.py \
  --dataset_name <dataset> \
  --UniBias <Option for using UniBias> \
  --Calibration <Option for evaluating calibration methods> \
  --seed <random_seed> \
  --format_index <Ggenerate prompts with different formats> \
  --order_index <Generate prompts with varying example orders>
```
---
## Acknowledgment

The code for calibration baselines evaluation is from [DC](https://github.com/fywalter/label-bias) and [PC](https://github.com/fywalter/label-bias). We appreciate their excellent contributions!

---
## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{zhou2024unibias,
  title={UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation},
  author={Zhou, Hanzhang  and
    Feng, Zijian and
    Zhu, Zixiao  and
    Qian, Junlang  and
    Mao, Kezhi},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

