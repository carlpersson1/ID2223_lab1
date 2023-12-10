# Fine-tuning Whisper for German transcription
## Training
The small Whisper model was fine-tuned with German audio/transcription pairs, 
according to the example notebook. The model was fine-tuned until it converged at around
40 WER on the test set, which was found to be unsatisfactory. In order to further increase
the WER, changes in hyperparameters was considered. Specifically learning rate, weight decay
and dropout was changed. The hyperparameters changed from:

learning rate from 1.e-5 to 1.25e-5,
weight decay from 0 to 1.e-6 and 
dropout from 0.1 to 0.05

A small amount of weight decay was added to prevent over-fitting from decreasing the dropout
and the learning rate was increased slightly to counteract some of the loss in training speed 
from the added weight decay. Finally, the dropout was halved since the original 10% 
dropout seemed fairly high for the small model. Continuing from the model that 
converged to 40 WER we found that the model decreased to ~32 WER with the new 
parameters, which is a solid decrease in the word error rate!

## Inference pipeline - Pronunciation practice
In order to effectively make use of the Whisper model a pronunciation practice app was
implemented. This is a fairly interesting application that can show the capabilities of
the Whisper model, the application was built using GradIO and is currently hosted on
a huggingface space: [Pronunciation practice app](https://huggingface.co/spaces/carlpersson/whisper).

## Ways to further improve the model performance
Some results of the applied improvements can be found in the training section!
### Model-centric approach
For this approach, the results and improvements used can be found above in the Training section.
### Data-centric approach
For the data-centric approach, the data could be extended with additional voice to text data, which specifically targets the
expected words that are going to be used. Since, in this case the app helps with pronounciation, it might be a good idea to use very difficult words
in the training data, that are rather hard to be pronounced. In German there exist a lot of longer words and uncommon pronounicaations. Focusing on those
improves the model specifically for the use of practicing pronounciation.

## Feature engineering pipeline and training pipeline
The notebook provided already contained most of the code that was needed to build a feature pipeline and
training pipeline. For the feature pipeline, all the code dealing with the preparation of data was grouped together.
The process simply starts by either fetching the dataset or if it is already stored in the feature store, loading it and preparing it.
The feature store is in this case a Google Drive space that is simply connected and loaded to the Colab environment. There is functionallity 
to store the downloaded dataset in the drive, as well as store the processed features. 
They might be very large, so both unprocesses and processed download options are provided. The feature pipline connects to the drive, loads the data,
processes it if not already done, and prepares it for the model to be used. 
The training pipeline consists of loading the model, preparing it, loading the weights if they
are found, and running training. In addition an evaluation is run to evaluate the model and optionally can be pushed to HuggingFace Hub, 
which functions as the model registry. The weights stored as checkpoints are also located in the drive, 
because checkpoints are by default stored in the given output folder found in the training arguments. 
Finally both feature and training pipeline can be run indepdently.
