## to train nn on large images 
epoch = 0


while epoch <2:
  i =0
  while i<130:
    k = 0
    X_train = []
    print('epoch : ',epoch)
    print('i : ', i)
    while k<100:
      name = 'drive/Colab Notebooks/train_images/Img-' + str(i*100 + k+1) +'.jpg'
      img = cv2.imread(name)
      img = cv2.resize(img, (500,500))
      X_train.append(img)
      k = k + 1
    print('loading 100 images done')
    X_train_ = np.asarray(X_train)
    y_train_ = y_train[i*100 : (i+1)*100]
  
    print(i*100 + k+1)
  
    print(i*100)
    print((i+1)*100)
    
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    model.fit_generator(gen.flow(X_train_, y_train_, batch_size = 50),
                    steps_per_epoch=len(X_train) /50,epochs=1,verbose = 1,
                    shuffle = True)
    model.save('resnet_500.h5')
    i = i + 1
  epoch = epoch + 1
