# Chukchi ASR

This repository contains experiments for developing an automatic speech recognition (ASR) system for the Chukchi language. The project focuses on low-resource ASR for a polysynthetic language and compares several multilingual pretrained models and fine-tuning strategies, with special attention to expedition/fieldwork recordings.

The work was carried out as part of the thesis project **“Development of a Neural Network System for Chukchi Language Speech Recognition.”**

## Project overview

The main goal of the project is to evaluate how modern multilingual ASR models can be adapted to Chukchi expedition speech domain under low-resource conditions. The experiments compare different ways of using available labeled data from several domains:

- **Chuklang expedition recordings** — the main target domain;
- **Radio Purga recordings** — radio/news speech;
- **Bible/IBT recordings** — read religious speech.

The experiments focus on two model families:

- **MMS** (`facebook/mms-1b-all`) with the Chukchi `ckt` adapter;
- **XLS-R** (`facebook/wav2vec2-xls-r-1b` and `facebook/wav2vec2-xls-r-300m`).

The main evaluation metrics are:

- **WER** — word error rate;
- **CER** — character error rate.

CER is especially important for Chukchi because long and morphologically complex wordforms make word-level evaluation less informative on its own.

## Repository structure

```text
chukchi_ASR/
├── notebooks/
│   ├── mms_adapter_*.ipynb
│   ├── xlsr_1b_*.ipynb
│   └── xlsr_300m_*.ipynb
│   ├── ckt_ASR.ipynb
│   └── Expedition_chukchi_ASR_pipeline.ipynb
│
├── logs/
│   └── *_log_history.csv
│
├── results.csv
└── README.md
```

## Experimental configurations

The repository contains notebooks for several experimental strategies based on MMS and XLS-R. The experiments differ in the model backbone, the training data, and the adaptation strategy.

### MMS experiments

| Notebook | Description |
|---|---|
| `mms_adapter_chuklang_only.ipynb` | Adapter fine-tuning of `facebook/mms-1b-all` with the Chukchi `ckt` adapter on Chuklang only train data |
| `mms_adapter_pooled.ipynb` | Adapter fine-tuning on pooled Chuklang + Radio + Bible train data |
| `mms_adapter_staged_bible+radio_chuklang.ipynb` | Two-stage training: Bible + Radio train data first, then Chuklang train data, the stage 1 checkpoint is selected on the target-domain Chuklang evaluation data |
| `mms_adapter_staged_bible+radio_chuklang_source_eval.ipynb` | Two-stage training where the stage 1 checkpoint is selected on the source-domain Bible + Radio evaluation data |

### XLS-R experiments

| Notebook | Description |
|---|---|
| `xlsr_1b_chuklang_only.ipynb` | Fine-tuning `facebook/wav2vec2-xls-r-1b` on Chuklang only train data |
| `xlsr_1b_pooled_to_chuklang+stage2_chuklang.ipynb` | Fine-tuning on pooled train data followed by additional Chuklang adaptation |
| `xlsr_1b_staged_bible_radio_to_chuklang_source_eval.ipynb` | Two-stage XLS-R 1B training: Bible + Radio first, then Chuklang, the stage 1 checkpoint is selected on the source-domain evaluation data |
| `xlsr_300m_staged_bible_radio_to_chuklang_source_eval.ipynb` | Two-stage XLS-R 300M training: Bible + Radio first, then Chuklang, the stage 1 checkpoint is selected on the source-domain evaluation data |

## Results

The file [`results.csv`](results.csv) contains the main evaluation results for the trained models. It includes model configuration names, checkpoint information, evaluation domains, and WER/CER values.

Training logs for individual experiments are stored in the [`logs`](logs/) directory.

## Best model

The best model by CER was selected from the experimental runs and saved separately for practical inference:

- Hugging Face model repository: [`tadgeis/mms-1b-all-ckt-best-cer-model`](https://huggingface.co/tadgeis/mms-1b-all-ckt-best-cer-model)

This repository contains only the files required to load the model and run inference. Intermediate checkpoints, optimizer states, and other training artifacts were removed to make the model repository more compact.

All trained models are available in public Hugging Face repositories under the author profile: [`https://huggingface.co/tadgeis`](https://huggingface.co/tadgeis)

## Practical use

The best model can be used for preliminary transcription of Chukchi expedition recordings. The end-to-end pipeline for applying the best ASR model to long expedition recordings is presented in [`notebooks/Expedition_chukchi_ASR_pipeline.ipynb`](notebooks/Expedition_chukchi_ASR_pipeline.ipynb). The pipeline includes audio conversion, VAD-based speech segmentation, ASR inference on segments, Cyrillic-to-Chuklang-IPA-like transliteration, and export of results to CSV and ELAN (`.eaf`) files.

The ASR output should be treated as a draft transcription rather than a final annotation. Manual verification is still required, especially for noisy recordings, overlapping speech, false starts, Russian insertions, and segmentation errors.

## Example: loading the best model

```python
import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC

model_id = 'tadgeis/mms-1b-all-ckt-best-cer-model'

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()
```

## Data

The original audio and transcript data are not included in this repository. The data are stored separately in a private Hugging Face dataset repository, since part of the material can only be shared with permission from the Chuklang project.

This GitHub repository contains only the code, training notebooks, training logs, and experiment-level results. The trained model checkpoints are available through public Hugging Face model repositories, while access to the underlying data is restricted.

## Citation

If you use this repository, please cite the corresponding thesis/project:

```bibtex
@misc{chukchi_asr_2026,
  title = {Development of a Neural Network System for Chukchi Language Speech Recognition},
  author = {Trenikhina, Taisiia},
  year = {2026},
  note = {GitHub repository: https://github.com/tadgeislamins/chukchi_ASR}
}
```

## Acknowledgements

This project builds on previous work on Chukchi ASR and multilingual low-resource speech recognition, including Safonova et al. (2022), MMS, and XLS-R.

Part of the data used in this project is associated with the Chuklang project. This work also builds on the dataset preparation described in Safonova et al. (2022), including the collection, verification, and preprocessing of the Radio Purga data, as well as preprocessing of the Chuklang expedition data.

I am grateful to my thesis supervisor and to everyone who contributed to the collection, preparation, and annotation of the Chukchi speech data used in this work.
