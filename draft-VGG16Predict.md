## VGG16

VGG16 is a convilutional neural network that won the ImageNetCompetition in xxxx. 

ToDo add more content

## VGG16 in DeepLearning4J

This Network was created by taking a pretrained network and the trained weights and importing it using DeepLEarning4J's
modelimport feature. 

## Live demo

Below is a form that you can use to select an image from your computer, that image will be uploaded and run 
through the neural network. The output will be the percent probability that the network applies 
to one of 1000 possible labels. You will see the top 5. 

## Interpreting the result

If you submit a picture of yourself, please note that this competition was not for facial recognition. 
The network will certainly fail to identify you, images of cats and dogs it will do much better with. 

## Live Demo

<!-- 
<iframe src="https://54.67.56.24/VGGpredict" width="400" height="300" style="display:block; margin: 0 auto;">&nbsp;</iframe>
-->

<iframe src="https://52.174.183.106/VGGpredict" width="400" height="300" style="display:block; margin: 0 auto;">test </iframe>

## Using VGG16 pretrained for your use case

Modifying this trained network for your use case would involve either "transfer learning" a feature we are working on, 
or loading the model and then performing additional training on your dataset. 



