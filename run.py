import tensorflow as tf
import matplotlib.pyplot as plt

from attack import input_gradient, fgsm

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

#beginning the attack , choosing one victim image
image=x_test[0]
label=y_test[0]
logits=model(image[None,...])
probs=tf.nn.softmax(logits)
print("true label:",label)
print("predicted:",int(tf.argmax(probs[0])),"confidence:", float(tf.reduce_max(probs)))

grad=input_gradient(model,image,label)
print(grad.shape)
plt.figure(figsize=(5,5))
plt.imshow(grad[0],cmap="seismic")
plt.title("gradient")
plt.savefig("figures/input_gradient.png",dpi=150,bbox_inches="tight")
plt.close()

epsilon=0.1
adv=fgsm(model,image,label,epsilon)
def predict(x):
  probs=tf.nn.softmax(model(x))
  return int(tf.argmax(probs[0])),float(tf.reduce_max(probs))
print("clean:",predict(image[None,...]))
print("adversarial:",predict(adv))
fig,ax=plt.subplots(1,3,figsize=(9,3))
ax[0].imshow(image,cmap="Blues")
ax[0].set_title("original")
ax[1].imshow(adv[0],cmap="Blues")
ax[1].set_title("adversarial")
ax[2].imshow(adv[0]-image,cmap="seismic")
ax[2].set_title("difference")
for a in ax:
  a.axis("off")
plt.savefig("figures/clean_vs_adversarial.png",dpi=150,bbox_inches="tight")
plt.close()
