# create the base pre-trained model

no_of_non_trainable_layers = 135

base_model =keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(500,500,3), pooling=max)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

for layer in base_model.layers[:no_of_non_trainable_layers]:
    layer.trainable = False
for layer in base_model.layers[no_of_non_trainable_layers:]:
    layer.trainable = True


# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x =Dropout(.5)(x)
x = Dense(512, activation='relu')(x)
x =Dropout(.5)(x)
# and a logistic layer -- let's say we have 30 classes
predictions = Dense(30, activation='softmax')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)


model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
model.summary()
