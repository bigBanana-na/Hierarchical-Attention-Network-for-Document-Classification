#coding=utf-8
import tensorflow as tf
import numpy as np
import time
import os
from dataLoad import read_dataset, batch_iter, batch
from model import HAN

# Data loading params
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
flags.DEFINE_integer("vocab_size", 46960, "vocabulary size")
flags.DEFINE_integer("num_classes", 5, "number of classes")
flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
flags.DEFINE_float("learning_rate", 0.01, "learning rate")
flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")

train_x, train_y, dev_x, dev_y = read_dataset()
print("data load finished")

han = HAN(vocab_size=FLAGS.vocab_size,
          num_classes=FLAGS.num_classes,
          embedding_size=FLAGS.embedding_size,
          hidden_size=FLAGS.hidden_size)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function(reduce_retracing=True)
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = han(x_batch, training=True)
        loss = loss_object(y_batch, predictions)
    gradients = tape.gradient(loss, han.trainable_variables)
    optimizer.apply_gradients(zip(gradients, han.trainable_variables))

    train_loss(loss)
    train_accuracy(y_batch, predictions)

@tf.function(reduce_retracing=True)
def test_step(x_batch, y_batch):
    predictions = han(x_batch, training=False)
    t_loss = loss_object(y_batch, predictions)

    test_loss(t_loss)
    test_accuracy(y_batch, predictions)

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

for epoch in range(FLAGS.num_epochs):
    print('Epoch %d/%d' % (epoch + 1, FLAGS.num_epochs))
    for batch_data in batch_iter(list(zip(train_x, train_y)), FLAGS.batch_size, 1):
        x_batch_raw = [x[0] for x in batch_data if x[0]]
        y_batch = np.array([x[1] for x in batch_data if x[0]])
        if len(x_batch_raw) == 0:
            continue
        x_batch, _, _ = batch(x_batch_raw)
        train_step(x_batch, y_batch)
    
    for batch_data in batch_iter(list(zip(dev_x, dev_y)), FLAGS.batch_size, 1):
        x_batch_raw = [x[0] for x in batch_data if x[0]]
        y_batch = np.array([x[1] for x in batch_data if x[0]])
        if len(x_batch_raw) == 0:
            continue
        x_batch, _, _ = batch(x_batch_raw)
        test_step(x_batch, y_batch)
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()