import tensorflow as tf

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape,x_train.dtype,x_train.min(),x_train.max())
print(y_train[:10])

x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model=tf.keras.Sequential([tf.keras.Input(shape=(28,28)),
                           tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(120,activation="relu"),
                           tf.keras.layers.Dense(10)])
model.summary()

losss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam",loss=losss,metrics=["accuracy"])
history=model.fit(x_train,y_train,epochs=5,batch_size=32,validation_split=0.1)

test_loss,test_acc=model.evaluate(x_test,y_test,verbose=1)
print("clean test accuracy:",test_acc)
model.save("mnist_model.keras")
