# LLM Diffusion Text-to-Audio Generation 

## Quickstart on Google Colab

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)] | (https://colab.research.google.com/drive/1J9WXds92CypFKAW7uyh05OyHqT7PCgQ5#scrollTo=Oxr4tYgWw1uL)




## Quickstart Guide

Download the model and generate audio from a text prompt:

```python
import IPython
import soundfile as sf
from tango import Tango

tango = Tango("declare-lab/tango2")

prompt = "An audience cheering and clapping"
audio = tango.generate(prompt)
sf.write(f"{prompt}.wav", audio, samplerate=16000)
IPython.display.Audio(data=audio, rate=16000)
```
[CheerClap.webm](https://user-images.githubusercontent.com/13917097/233851915-e702524d-cd35-43f7-93e0-86ea579231a7.webm)

The model will be automatically downloaded and saved in cache. Subsequent runs will load the model directly from cache.

The `generate` function uses 100 steps by default to sample from the latent diffusion model. We recommend using 200 steps for generating better quality audios. This comes at the cost of increased run-time.

```python
prompt = "Rolling thunder with lightning strikes"
audio = tango.generate(prompt, steps=200)
IPython.display.Audio(data=audio, rate=16000)
```
[Thunder.webm](https://user-images.githubusercontent.com/13917097/233851929-90501e41-911d-453f-a00b-b215743365b4.webm)

<!-- [MachineClicking](https://user-images.githubusercontent.com/25340239/233857834-bfda52b4-4fcc-48de-b47a-6a6ddcb3671b.mp4 "sample 1") -->

Use the `generate_for_batch` function to generate multiple audio samples for a batch of text prompts:

```python
prompts = [
    "A car engine revving",
    "A dog barks and rustles with some clicking",
    "Water flowing and trickling"
]
audios = tango.generate_for_batch(prompts, samples=2)
```
This will generate two samples for each of the three text prompts.

More generated samples are shown [here](https://github.com/declare-lab/tango/blob/master/samples/README.md).

## Prerequisites

Our code is built on pytorch version 1.13.1+cu117. We mention `torch==1.13.1` in the requirements file but you might need to install a specific cuda version of torch depending on your GPU device type.

Install `requirements.txt`.

```bash
git clone https://github.com/declare-lab/tango/
cd tango
pip install -r requirements.txt
```

You might also need to install `libsndfile1` for soundfile to work properly in linux:

```bash
(sudo) apt-get install libsndfile1
```

## Datasets

Follow the instructions given in the [AudioCaps repository](https://github.com/cdjkim/audiocaps) for downloading the data. The audio locations and corresponding captions are provided in our `data` directory. The `*.json` files are used for training and evaluation. Once you have downloaded your version of the data you should be able to map it using the file ids to the file locations provided in our `data/*.json` files.

## How to train?
We use the `accelerate` package from Hugging Face for multi-gpu training. Run `accelerate config` from terminal and set up your run configuration by the answering the questions asked.

You can now train on the AudioCaps dataset using:

```bash
accelerate launch train.py \
--text_encoder_name="google/flan-t5-large" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

The argument `--augment` uses augmented data for training as reported in our paper. We recommend training for at-least 40 epochs, which is the default in `train.py`.

To start training from our released checkpoint use the `--hf_model` argument.

```bash
accelerate launch train.py \
--hf_model "declare-lab/tango" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder --augment --snr_gamma 5 \
```

Check `train.py` and `train.sh` for the full list of arguments and how to use them.

The training script should automatically download the AudioLDM weights from [here](https://zenodo.org/record/7600541/files/audioldm-s-full?download=1). However if the download is slow or if you face any other issues then you can: i) download the `audioldm-s-full` file from [here](https://huggingface.co/haoheliu/AudioLDM-S-Full/tree/main), ii) rename it to `audioldm-s-full.ckpt`, and iii) keep it in `/home/user/.cache/audioldm/` direcrtory.

To train  on the Audio-alpaca dataset from  checkpoint using:
The training script will download audio_alpaca wav files and save it in {PATH_TO_DOWNLOAD_WAV_FILE}/audio_alpaca. Default location will be ~/.cache/huggingface/datasets.
```bash
accelerate launch  tango2/tango2-train.py --hf_model "declare-lab/tango-full-ft-audiocaps" \
--unet_model_config="configs/diffusion_model_config.json" \
--freeze_text_encoder  \
--learning_rate=9.6e-7 \
--num_train_epochs=5  \
--num_warmup_steps=200 \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4  \
--gradient_accumulation_steps=4 \
--beta_dpo=2000  \
--sft_first_epochs=1 \
--dataset_dir={PATH_TO_DOWNLOAD_WAV_FILE}
```


## How to make inferences?

### From your trained checkpoints

Checkpoints from training will be saved in the `saved/*/` directory.

To perform audio generation and objective evaluation in AudioCaps test set from your trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
--original_args="saved/*/summary.jsonl" \
--model="saved/*/best/pytorch_model_2.bin" \
```

Check `inference.py` and `inference.sh` for the full list of arguments and how to use them.

To perform audio generation and objective evaluation in AudioCaps test set for  :

```bash
CUDA_VISIBLE_DEVICES=0 python tango2/inference.py \
--original_args="saved/*/summary.jsonl" \
--model="saved/*/best/pytorch_model_2.bin" \
```



