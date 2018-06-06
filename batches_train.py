## to train nn on large images 

##hyperparameters to be tunes according to the dataset

no_of_epochs = 2
image_height = 500
image_width = 500
no_of_small_batches = 130

## actual code
epoch = 0
while epoch <no_of_epochs:
  i =0
  while i<no_of_small_batches:
    k = 0
    X_train = []
    print('epoch : ',epoch)
    print('i : ', i)
    while k<100:
      name = 'drive/Colab Notebooks/train_images/Img-' + str(i*100 + k+1) +'.jpg'
      img = cv2.imread(name)
      img = cv2.resize(img, (image_height, image_width))
      X_train.append(img)
      k = k + 1
    print('loading 100 images done')
    X_train_ = np.asarray(X_train)
    y_train_ = y_train[i*100 : (i+1)*100]
  
    #fitting the model 
    model.fit_generator(gen.flow(X_train_, y_train_, batch_size = 50),
                    steps_per_epoch=len(X_train) /50,epochs=1,verbose = 1,
                    shuffle = True)
    # saving the model
    model.save('resnet_500.h5')
    i = i + 1
  epoch = epoch + 1
