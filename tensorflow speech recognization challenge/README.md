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





