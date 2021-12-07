# train a generative adversarial network on a one-dimensional function
import pandas as pd
import numpy as np
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import IPython


# define the standalone discriminator model
def define_discriminator(n_inputs=4):
    model = Sequential()
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_outputs=4):
    model = Sequential()
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# generate n real samples with class labels
def generate_real_samples(df, half_batch_size):
    X = np.zeros((half_batch_size, 4))
    for j in range(3):
        force = f'force_{j}'
        X[:, j] = df[force]
    X[:, 3] = df['tools_angle']
    # generate class labels
    y = ones((half_batch_size, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n, 1))
    return X, y


# evaluate the discriminator and plot real and fake points
def summarize_performance(df, half_batch_size, epoch, generator, discriminator, latent_dim):
    # prepare real samples
    x_real, y_real = generate_real_samples(df, len(df))

    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, len(df))
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    x_total = np.concatenate((x_real, x_fake), axis=0)
    y_total = np.concatenate((y_real, y_fake), axis=0)

    predictions = np.round(discriminator.predict(x_total))
    accuracy = accuracy_score(y_total, predictions, normalize=True)
    # summarize discriminator performance
    print(f'Epoch:{epoch} Accuracy:{accuracy}')

    # scatter plot real and fake data points
    pyplot.figure(figsize=(5, 5))
    pyplot.scatter(x_real[:, -1], x_real[:, -1], color='red', label='Real')
    pyplot.scatter(x_real[:, -1], x_fake[:, -1], color='blue', label='Fake')
    pyplot.legend()
    pyplot.show()


# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, df_full, df_test, n_epochs=10000, n_batch=2048, n_eval=4096):
    # manually enumerate epochs
    for i in range(n_epochs):
        df = df_full
        print('EPOCH: ', i)
        # determine half the size of one batch, for updating the discriminator
        half_batch = int(n_batch / 2)
        old_batch = 0

        for batch in range(half_batch, len(df), half_batch):
            # prepare real samples
            x_real, y_real = generate_real_samples(df.iloc[old_batch:batch], half_batch)
            # prepare fake examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # update discriminator
            d_model.train_on_batch(x_real, y_real)
            d_model.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            gan_model.train_on_batch(x_gan, y_gan)

            old_batch = batch

        # evaluate the model every n_eval epochs
        summarize_performance(df_test, half_batch, i, g_model, d_model, latent_dim)


# size of the latent space
latent_dim = 5
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)

# Get data
df_full = pd.read_csv('logs.csv').sample(frac=1)
df_test = df_full.iloc[int(0.8*len(df_full)):-1]
df_train = df_full.iloc[0:int(0.8*len(df_full))]

# train model
train(generator, discriminator, gan_model, latent_dim, df_full=df_train, df_test=df_test)