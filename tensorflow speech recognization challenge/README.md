# audio classification challenge 
this is my solution approach to kaggle challenge <a href = 'https://www.kaggle.com/c/tensorflow-speech-recognition-challenge'>link</a><br/>
download the data from the link above<br/>

i am using librosa library in python to extract features from audio clips <a href = 'https://librosa.github.io/librosa/'> click here </a> for more information <br/>

here i am using five different features feel free to experiment with different avilable features <br/>


what i have considered for this model:<br/>
1. 
accuracy of the validation set was very poor using only the audio samples provided. so we need to have some kind of audio augmentation that our model can perform better on the dev set<br/>
for this time shifting,  speed tunning as well as mixing the base audio with different background noises provided with this data set<br/>
doing this i got a significant improvement in the dev set <br/>

2. model should not be too simple or too complex both comes with different problem so make the fairly complex which using some of the regularising tools like using dropout<br/>

NOTE: <br/>
achieved an acuracy of of 77% on the validation set and 85% on the test set. on the confusion matrix we correctly able to classify all the classes 70 percent of the time (consider there are 30 classes) so this result is not bad . further this result can be imporved by using more forms of audio augmentation and adding more background noises with the help of the function extract_feature_fixed_background_noise. i tried this but not able to complete it as it may take almost a day on my computer to do it. feel free to use my model which is simple model without augmentation or experiment with your own model and comment if any probelm is found



