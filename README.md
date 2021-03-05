# DNNSeg

DNNSeg is a repository for unsupervised speech segmentation and classification using deep neural frameworks.
You are currently in the `dnnseg_german` branch, where I'm working on my Masters thesis. 
This branch was forked from [Cory Shain's Repository](https://github.com/coryshain/dnnseg/tree/NAACL19), where the results of Shain & Elsner (2019) can be reproduced.

To run this script, start with
`python -m dnnseg.build <PATH-TO-METADATA> <PATH-TO-WORD-INFORMATION> -d <PATH-TO-DE2> -s <PATH-TO-SP1>`

Once the data have been preprocessed, models can be trained by running:

`python -m dnnseg.bin.train <PATH-TO-INI-FILE>`

The models are defined in the following files at the repository root:

  - `german_classify.ini`
  - `german_classify_nospeaker.ini`
  - `german_classify_nobsn.ini`
  - `german_classify_nospeaker_nobsn.ini`

# Metadata

Metadata: folder containing
- `german_files.txt` : a list of all .wav files to be read. No header necessary.
- `german_vad.txt` : a list of .wav files, audio start point in ms, and audio end point in ms. No headers necessary

Word information: folder containing
- `german.vad` : a list of .wav files, audio start point in ms, audio end point in ms, and label. No headers necessary.

## References

* Shain, Cory and Elsner, Micha (2019). Measuring the perceptual availability of phonological features during language
  acquisition using unsupervised binary stochastic autoencoders. _NAACL19_.
