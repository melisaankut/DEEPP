# Classification of Paintings from Different Perspectives

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)


## Abstraction

- Every artist is inspired by different artists and is
curious about the artists s/he likes. Here we will
try to predict which trends and styles of famous
artists resemble the drawing to be analyzed. We
want to help artists to motivate themselves, learn
more about their styles, etc. We do experiment
on a dataset with 20000 images and 10 features.
Our goal is to make style predictions on images,
based on their specifications. We use CNN to train
our model. We will cover the technical details in
the following sections. However, we must say
that many features should be considered in this
classification. This greatly affects the accuracy
rate. We will tell you what we have done on this
subject and what those who want to work on this
subject can do.

## Introduction
Throughout history, people have created many visual artworks,
starting with murals and continuing today as electronic
paintings. During this period, many movements and
artists emerged. While these movements sometimes have
very different styles, sometimes they can be difficult to distinguish.
Especially people far from the art history may
have difficulties in this regard. A new painter may not know
exactly which art movement or artist he is close to. If there
is no one to guide them at this point, they may experience
some difficulties. For example, this information is very important
in terms of knowing other artists that he can take as
an example. 

In this regard, we wanted to develop a machine
learning model in which one can learn to what trend his or
her picture is close to. Our goal is for the classifier to be
able to predict about 10 different art movements. Therefore,
we are planning to use CNN.We’ll talk about the technical
details later, but of course we do not claim that this method
is the best way. With progress in this area, much more complex
structures can be built and a much higher accuracy can
be achieved. We believe that much more work needs to be
done and will be done in this regard.

## The Approach

- [Dataset] - We found a large dataset compiled from WikiArt
for a Kaggle competition. It has more than 100,000 paintings
belonging to 2,300 artists, covering a lot of styles and
eras.
- Convolutional Neural Networks 
- Convolutional Autoencoders
- Multi-Class Cross-Entropy Loss
- Optimization Algorithm


## Dataset Link For .hkl Files

- [.hkl files] - The parts we put in the comment line in our code are optional parts. We carried out our work with hickle files that we had read and created before. 

## Additional Work
Towards the end of the project, it also made sense to take advantage of the TPU offered to us by Colab. We have kept the implemented version as a comment line for future use in the development phase of the code.
## Project Introduction Video
- Also, at the end of the project, we created a short video to introduce our work : 

[![](https://img.youtube.com/vi/URuNFdCzMnI/0.jpg)](https://www.youtube.com/watch?v=URuNFdCzMnI)

## Authors
+ [Melisa Ankut](https://www.linkedin.com/in/melisa-ankut-49003217b/)
+ [Ertürk Gürses](https://www.linkedin.com/in/erturkgurses/)

## License

MIT


   [Dataset]: https://www.kaggle.com/c/painter-by-numbers/data
   [.hkl files]: https://drive.google.com/drive/folders/1dDHOJhj6NZmKm72-_ujA5itwHhbvJLTE


