library(keras)

fashion_mist<- dataset_fashion_mnist()

c(train_images,train_labels) %<-% fashion_mist$train
c(test_images,test_labels) %<-% fashion_mist$test

#Each image is mapped to a single label. Since the class names are not included with the dataset,
#we'll store them in a vector to use later when plotting the images.

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

dim(train_images)

dim(train_labels)

train_labels[1:20]

library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1,aes(x=x,y=y,fill = value))+
  geom_tile()+
  scale_fill_gradient(low="white",high="black",na.value = NA)+
  scale_y_reverse()+theme_minimal()+
  theme(panel.grid = element_blank())+
  theme(aspect.ratio = 1)+
  xlab("")+
  ylab("")

#We scale these values to a range of 0 to 1 before feeding to the neural network model. For this, we simply divide by 255.
train_images <- train_images / 255
test_images <- test_images / 255

#Display the first 25 images from the training set and display the class name below each image. 
#Verify that the data is in the correct format and we're ready to build and train the network.

par(mfcol=c(5,5))
par(mar=c(0,0,1.5,0),xaxs='i',yaxs='i')

for(i in 1:25){ 
  img<- train_images[i, , ]
  img<- t(apply(img,2,rev))
  image(1:28,1:28,img,col=gray((0:255)/255),xart='n',main=paste(class_names[train_labels[i]+1]))
}

#building model 
#setting up the layers 
model<-keras_model_sequential()
model%>%
  #layer_flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels. 
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

#compile model
#Loss function - This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
#Optimizer - This is how the model is updated based on the data it sees and its loss function.
#Metrics -Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

model%>%compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics = c('accuracy')
                )

#train model 
model %>% fit(train_images, train_labels, epochs = 5)

#evaluate the accuracy
score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

predictions <- model %>% predict(test_images)

#A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to each of the 10 
#different articles of clothing. We can see which label has the highest confidence value:
predictions[1, ]

which.max(predictions[1, ])

#Alternatively, we can also directly get the class prediction:
  
class_pred <- model %>% predict_classes(test_images)

#predicted class
class_pred[1:20]

#actual class
test_labels[1:20]

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')

for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}

#use the trained model to make a prediction about a single image.

# Grab an image from the test dataset
# take care to keep the batch dimension, as this is expected by the model
img <- test_images[1, , , drop = FALSE]
dim(img)

#Now predict the image:
predictions <- model %>% predict(img)
predictions

# subtract 1 as labels are 0-based
prediction <- predictions[1, ] - 1
which.max(prediction)

#Or, directly getting the class prediction again:
class_pred <- model %>% predict_classes(img)
class_pred