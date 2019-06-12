# Sequence translation: text-to-speech

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Stages of project:
- [x] Find dataset
- [ ] Data pre-processing
    - [ ] Text pre-processing
    - [ ] Audio pre-processing
- [ ] Sequence model architecture
    - [x] Choose state of the art architecture - **Tacotron**
    - [ ] Create architecture of the model from the paper
- [ ] Training
    - [ ] Implement train module
    - [ ] Train model
    - [ ] Add tensorboard
- [ ] Evaluation
    - [ ] Inference module
    - [ ] Benchmarking

## Model Architecture
![tacotron architecture diagram](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/a072c2a400f62f720b68dc54a662fb1ae115bf06/2-Figure1-1.png)



## Dependencies

### Project using such dependencies:
- 1


### You can install the latest dependencies:
  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/numpy/badges/version.svg)](https://anaconda.org/anaconda/numpy): `conda install numpy`
  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/scipy/badges/version.svg)](https://anaconda.org/anaconda/scipy): `conda install scipy`
  - [![Anaconda-Server Badge](https://anaconda.org/pytorch/pytorch/badges/installer/conda.svg)](https://conda.anaconda.org/pytorch): `conda install pytorch torchvision -c pytorch`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/tqdm/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge): `conda install -c conda-forge tqdm`
  - [![PyPI version](https://badge.fury.io/py/tensorboardX.svg)](https://badge.fury.io/py/tensorboardX): `pip install tensorboardX`


## Literature and references:
- Tacotron: Towards End-to-End Speech Synthesis	[arXiv:1703.10135](https://arxiv.org/abs/1703.10135) [cs.CL]
- (The LJ Speech Dataset)[https://keithito.com/LJ-Speech-Dataset/]
- [Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html) @ [Dive into Deep Learning](https://d2l.ai/index.html) interactive book
- [Text to Speech Deep Learning Architectures](http://www.erogol.com/text-speech-deep-learning-architectures/)
- [Deep Learning for Audio](http://slazebni.cs.illinois.edu/spring17/lec26_audio.pdf) Y. Fan, M. Potok, C. Shroba
- (Deep Learning for Text-to-Speech Synthesis, using the Merlin toolkit)[http://www.speech.zone/courses/one-off/merlin-interspeech2017/]
- (Babble-rnn: Generating speech from speech with LSTM networks)[http://babble-rnn.consected.com/docs/babble-rnn-generating-speech-from-speech-post.html]
- https://github.com/r9y9/tacotron_pytorch
- https://github.com/keithito/tacotron
- https://github.com/Kyubyong/tacotron
- [The Centre for Speech Technology Research](http://www.cstr.ed.ac.uk/)
- [Preparing Data for Training an HTS Voice](http://www.cs.columbia.edu/~ecooper/tts/data.html)
- [awesome speech synthesis/recognition papers](http://rodrigo.ebrmx.com/github_/zzw922cn/awesome-speech-recognition-speech-synthesis-papers)
