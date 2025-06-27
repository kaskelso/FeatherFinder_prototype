# FeatherFinder_prototype

## This repo is the result of learning machine learning and my wife asking me "whats bird is this feather from?"

I study birds and I'm often asked by people who found a feather what bird it belongs to and I often just shrug. I wanted to see if I could prototype a very small ML model to assign feathers to a given species. To do this I used a very small (less than 50 images) of feathers from online using the web crawler code and the USFWS feather atlas of Northern Goshawks, Great-horned owls, and Osprey. 

# Here is the order of operations:

# Feather image splitter

First I needed to split up the USFWS feather atlas images to just photos of single feathers. I couldnt find anything that did that well so I used a pre-trained model, Detectron2, to tune it on a small dataset (less than 60) images of random feathers. This actually worked great and could effectively split my photos up. 

# Train Resnet18 on feathers of the three species

Next I used Resnet18 to train on the dataset. The most important bit was freezing all the model layers expect the last so they didn't updating during training since I wanted to use the benefits of it being pretrained. I ended up getting ~96% trainging accuracy and 100% validation accuracy.

# Random tests

Now heres the important bit. It doesn't work well lol. Which is expected! Super small model just and this was just something I messed around with in one day. The model will try and classiy random images and if given a feather of a random bird will assign high probability to one of the three. I think an immediate benefit would be to use the tuned Detectron2 feather model to filter photos that don't have a feather in them. 
