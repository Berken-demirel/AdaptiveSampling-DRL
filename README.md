# AdaptiveSampling-DRL

## Initial Setup
## Running Scripts

## Motivational Example
#### The model architecture is implemented in [models.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/dana_MIT_constant.py). Requirements_2 is needed to run the script without errors.
<img src="./Figures/mot_jbhi.jpg" width="800">

## Model Architecture
#### The model architecture is implemented in [models.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/models.py). Requirements_2 is needed to run the script without errors.
<img src="./Figures/jbhi_arch.png" width="800">

## Utilities Script
#### [swd_utils.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/swd_utils.py) contains most of the important functions such as estimating the Multitaper PSD, configuration function provides Leave N One Out [Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), metric calculation etc. All functions are explained inside the script just after the definition. Feel free to investigate further.
<img src="./Human/img/crossval.gif" width="300">

## Reading the MIT-BIH data
#### Since the data got from [TUSZ Corpus](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tusz) is not ready to feed into our neural network directly, some file handling functions are coded in [read_TUSZ.py](https://github.com/Berken-demirel/SWD_Detect/blob/master/Human/read_TUSZ.py). The functioncan be generalized for further applications and not only limited to absz seizure inside the corpus. There are a lot of arguments that can modify the main purpose of the functions. Researchers are free to use our functions according to their needs by citing our paper. Please note that [pyedflib](https://pyedflib.readthedocs.io/en/latest/contents.html) is necessary to run this script. We will be using only two montages of their multi channel EEG data as shown in the Figure:
<img src="./Human/img/skull.png" width="300">

