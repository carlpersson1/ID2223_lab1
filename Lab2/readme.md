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
### Data-centric approach

## Feature engineering pipeline and training pipeline