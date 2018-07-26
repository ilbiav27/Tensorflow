
# Tensorflow
  It's written by Tensorflow
# Vgg
# Transfer Learning

  I relearn VGG16 with 5 classes (person, cat, dog, car, boat)
  you can learn your model with pretrained model.
  This code just transforms fully-connected layer with convolution layer.
  But, You can change even deep CNN layer.
  
# Prepare

  1. What you need to prepare is pictures as many as you can that you want to learn and test.
    I classify them with folders that named like no, n1, n2, ...
  2. Text File for shuffling pictures.
    That text file consist of lines.
    Each line shows all picture's location and class in random order.
  3. 'TL_ckpt' directory for saving check point.

# Procedure

  If you are Ready for Running,

1. Run 'TLvgg16_run.ipynb' by the time you want to quit.
  A check point file will be saved per 10 epoch.
2. Run test file.
  Check your checkpoint file name and write it down in test file.
  Put validation pictures
  See results and evaulate your model.
