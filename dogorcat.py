import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
from random import randint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from scipy import ndimage
from scipy import misc
import random
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


targets = []
features = []

files = glob.glob('train/*.jpg')

# for file in files[:500]:
for file in files:
    features.append(np.array(Image.open(file).resize((75, 75))))
    target = [1, 0] if "cat" in file else [0, 1]
    targets.append(target)


features = np.array(features)
targets = np.array(targets)

# print("features shape", features.shape)
# print("Targets shape", targets.shape)


# for a in [randint(0, len(features)) for _ in range(10)]:
#     plt.imshow(features[a], cmap="gray")
#     plt.show()

# train

X_train, X_valid, y_train, y_valid = train_test_split(
    features, targets, test_size=0.05, random_state=42)


# Création du modèle


# Placeholder
x = tf.placeholder(tf.float32, (None, 75, 75, 3), name="x")
y = tf.placeholder(tf.float32, (None, 2), name="y")
dropout = tf.placeholder(tf.float32, (None), name="dropout")


def create_conv(prev, filter_size, nb):
    # First convolution
    conv_W = tf.Variable(tf.truncated_normal(
        shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    conv_b = tf.Variable(tf.zeros(nb))
    conv = tf.nn.conv2d(prev, conv_W, strides=[
                        1, 1, 1, 1], padding='SAME') + conv_b
    # Activation: relu
    conv = tf.nn.relu(conv)
    # Pooling
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[
                          1, 2, 2, 1], padding='SAME')
    return conv


conv = create_conv(x, 8, 32)
conv = create_conv(conv, 5, 64)
conv = create_conv(conv, 5, 128)
conv = create_conv(conv, 5, 256)
conv = create_conv(conv, 5, 215)

flat = flatten(conv)
print(flat, flat.get_shape()[1])

# First fully connected layer
fc1_W = tf.Variable(tf.truncated_normal(shape=(int(flat.get_shape()[1]), 512)))
fc1_b = tf.Variable(tf.zeros(512))
fc1 = tf.matmul(flat, fc1_W) + fc1_b

# Activation.
fc1 = tf.nn.relu(fc1)

#fc1 = tf.nn.dropout(fc1, keep_prob=dropout)

# Last layer: Prediction
fc3_W = tf.Variable(tf.truncated_normal(shape=(512, 2)))
fc3_b = tf.Variable(tf.zeros(2))
logits = tf.matmul(fc1, fc3_W) + fc3_b

softmax = tf.nn.softmax(logits)

# Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

# Accuracy
predicted_cls = tf.argmax(softmax, axis=1)
correct_prediction = tf.equal(predicted_cls, tf.argmax(y, axis=1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
training_operation = optimizer.minimize(loss_operation)

# Train the model (depend of power of pc max 500 with a good computer)
# batch_size = 1
batch_size = 100
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# restore sess
saver = tf.train.Saver()
saver.restore(sess, "./saved-model/saved-model.ckpt")


def augmented_batch(batch):
    """
    """
    n_batch = []

    for img in batch:
        if random.uniform(0, 1) > 0.75:
            process_img = Image.fromarray(
                np.uint8(img.reshape(75, 75, 3))).rotate(randint(-45, 45))
            n_img = np.array(process_img)
            n_batch.append(n_img.reshape(75, 75, 3))
        else:
            n_batch.append(img)

    return n_batch


i = 0
for epoch in range(0, 10000):
    print(">> Epoch: %s" % epoch)
    # Shuffle
    indexs = np.arange(len(X_train))
    np.random.shuffle(indexs)
    X_train = X_train[indexs]
    y_train = y_train[indexs]

    for b in range(0, len(X_train), batch_size):
        batch = augmented_batch(X_train[b:b+batch_size])
        #batch = X_train[b:b+batch_size]

        if i % 20 == 0:
            #print(sess.run(predicted_cls, feed_dict={dropout: 1.0, x: batch, y: y_train[b:b+batch_size]}))
            print("Accuracy [Train]:", sess.run(accuracy_operation, feed_dict={
                  dropout: 1.0, x: batch, y: y_train[b:b+batch_size]}))
        sess.run(training_operation, feed_dict={
                 dropout: 0.8, x: batch, y: y_train[b:b+batch_size]})
        i += 1
    # save model trained

    save_path = saver.save(sess, "./saved-model/saved-model.ckpt")

    if epoch % 2 == 0:
        accs = []
        for b in range(0, len(X_valid), batch_size):
            accs.append(sess.run(accuracy_operation, feed_dict={
                        dropout: 1., x: X_valid[b:b+batch_size], y: y_valid[b:b+batch_size]}))
        print("Accuracy [Validation]", np.mean(accs))
