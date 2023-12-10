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
In order to effectively make use of the Whisper model, a pronunciation practice app was
implemented. This is a fairly interesting application that can show the capabilities of
the Whisper model, the application was built using GradIO and is currently hosted on
a huggingface space: [Pronunciation practice app](https://huggingface.co/spaces/carlpersson/whisper).

## Ways to further improve the model performance
Some results of the applied improvements can be found in the training section!
### Model-centric approach
A typical model-centric approach to increasing model performance is to execute a
hyperparameter search. This involves choosing which variables to perform the search on
such as learning rate, dropout, weight decay and so on. Then choosing a interval of values
to search each value for and evaluating the model at each combination of hyperparameters, 
while picking the model that performs the best. Adding more variables and searching the same amount of points
increases the computational need exponentially since for each value of the learning rate
every point in the interval chosen for the weight decay will have to be searched, and so on.
This can be incredibly computationally expensive. We did not opt for this approach.

Furthermore, another way of improving the model performance can be done through
architectural changes of the model. This could involve adding extra layers on top of the
model and training them to the new task while keeping the old pre-trained layers frozen.
Such an approach would likely keep the generalizability of the original model while allowing
it to perform another task. Another way of doing this is through LoRA, which could
be used to inject matrices parallel to some of the pre-trained matrices and only 
training the injected matrices with fewer parameters.

### Data-centric approach
For the data-centric approach there are multiple ways of increasing model performance. 
This involves cleaning the data for potentially bad inputs, or audio clips that lack any
transcriptions and so on. Properly, pre-processing the audio data can also help increase
model performance, this could involve normalizing the inputs to be normally distributed around
0 with a mean of 1, or clamping the values between -1 and 1. Also collecting more data 
is also quick and easy way to improve performance and consistency of the model, assuming
that the data is of good enough quality.

It is also important to consider the model use case. If the model is to be used by people 
in a more relaxed language, we want the input data to reflect that. We basically want
the training data distribution to be as close as possible to the input distribution we
expect to be using the model on. This could involve making sure that common phrases we
expect to be used as input to the model are also common in the training data. For our
pronunciation practice app it might be a good idea to use very difficult words
in the training data, that are rather hard to be pronounced. In German there exist a lot of
longer words and uncommon pronunciations. Focusing on those
improves the model specifically for the use of practicing pronunciation.

## Feature engineering pipeline and training pipeline
The notebook provided already contained most of the code that was needed to build a feature pipeline and
training pipeline. For the feature pipeline, all the code dealing with the preparation of data was grouped together.
The process simply starts by either fetching the dataset or if it is already stored in the feature store, loading it and preparing it.
The feature store is in this case a Google Drive space that is simply connected and loaded to the Colab environment. There is functionality 
to store the downloaded dataset in the drive, as well as store the processed features. 
They might be very large, so both unprocessed and processed download options are provided. The feature pipline connects to the drive, loads the data,
processes it if not already done, and prepares it for the model to be used. 
The training pipeline consists of loading the model, preparing it, loading the weights if they
are found, and running training. In addition, an evaluation is run to evaluate the model and optionally can be pushed to HuggingFace Hub, 
which functions as the model registry. The weights stored as checkpoints are also located in the drive, 
because checkpoints are by default stored in the given output folder found in the training arguments. 
Finally, both feature and training pipeline can be run independently.
