# deep learning method for image inpainting
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time

from IPython import display

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


def data_preprocess():
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset


def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model


def generate_test_image(show=False):
    test_generator = generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = test_generator(noise, training=False)

    if show:
        plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        plt.show()
    return generated_image


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# test_discriminator = discriminator()
# test_model = generate_test_image()
# result = test_discriminator(test_model)
# print(result)

# cross-entropy v.s. MSE:
# they are both good for classification problem, especially binary classification
# MSE with softmax might have very slow rate of convergence at the training starts
# while cross-entropy performs well and smooth


def G_loss(output):
    # generator loss is simply the cross entropy between prediction and 1
    # it represents how well the generator "cheated" the discriminator
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(output), output)


def D_loss(real, fake):
    # real represents the prediction of discriminator over real images
    # fake represents the prediction of discriminator over fake images
    # total loss is the sum of cross entropy of real and fake
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.ones_like(fake), fake)
    return real_loss + fake_loss


def train_step(images, G_opt, D_opt, noise_dim=100):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generator = generator_model()
        discriminator = discriminator_model()
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = G_loss(fake_output)
        dis_loss = D_loss(real_output, fake_output)
    G_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    D_gradient = dis_tape.gradient(dis_loss, generator.trainable_variables)
    G_opt.apply_gradients(zip(G_gradient, generator.trainable_variables))
    D_opt.apply_gradients(zip(D_gradient, discriminator.trainable_variables))
    return generator, discriminator


def train(dataset, epochs):
    G_opt = tf.keras.optimizers.Adam(1e-4)
    D_opt = tf.keras.optimizers.Adam(1e-4)
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            G, D = train_step(image_batch, G_opt, D_opt, 100)
            display.clear_output(wait=True)
            save_image(G, epoch + 1, seed)
    display.clear_output(wait=True)


def save_image(model, epoch, test):
    pred = model(test, training=False)
    pic = plt.figure(figsize=(4, 4))
    for i in range(pred.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pred[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()
