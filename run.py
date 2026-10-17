import tensorflow as tf
import numpy as np
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

#multiple images now
images=x_test[:1000]
labels=y_test[:1000]
images = np.expand_dims(x_test[:1000], axis=-1)
def accuracy_under_attack(model,images,labels,epsilon):
  adv=fgsm(model,images,labels,epsilon)
  preds=tf.argmax(model(adv),axis=1)
  return tf.reduce_mean(tf.cast(preds==labels,tf.float32))

epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies=[]
for eps in epsilons:
  acc=accuracy_under_attack(model,images,labels,eps)
  accuracies.append(acc)
  print(eps,np.round(acc,4))
plt.figure(figsize=(6,4))
plt.plot(epsilons,accuracies,marker="o")
plt.xlabel("epsilon")
plt.ylabel("accuracy")
plt.title("MNIST classifier under FGSM attack")
plt.grid(True,alpha=0.3)
plt.savefig("figures/accuracy_vs_epsilon.png",dpi=150,bbox_inches="tight")
plt.close()
