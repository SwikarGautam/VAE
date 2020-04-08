import ConvoNet.ConvoNet as cn
import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('data.pickle', 'rb') as f:
    train_data = pickle.load(f)  # training set of shape (70000, 784)

in_shape = 784
hid_shape = 256
lat_shape = 20
out_shape = in_shape

epoch = 10
mb_size = 64
lr = 1e-3


class VAE:

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def reconstruct(self, img):
        para = self.encoder.predict(img)
        mu, logvar = np.hsplit(para, 2)

        eps = np.random.randn(*mu.shape)
        latent = eps*np.exp(logvar/2) + mu

        out = self.decoder.predict(latent)
        return out

    def generate(self, latent_vec):
        return decoder.predict(latent_vec)

    def train(self, data):

        para = self.encoder.predict(data)
        mu, logvar = np.hsplit(para, 2)

        eps = np.random.randn(*mu.shape)
        latent = eps*np.exp(logvar/2)+mu

        KL_loss = 0.5*np.sum(np.exp(logvar) + mu**2 - logvar - 1)
        out = self.decoder.predict(latent)
        BCE = (-data*np.log(out)-(1-data)*np.log(1-out)).sum()

        decoder_data = list(zip(latent, data))

        dx = decoder.train(decoder_data, 1, lr, mb_size, False, ret_delta=True)

        d_mu = dx + mu
        d_logvar = 0.5*(dx*(latent-mu) + (np.exp(logvar) - 1))
        delta = np.hstack((d_mu, d_logvar))

        encoder_data = list(zip(data, delta))
        encoder.train(encoder_data, 1, lr, mb_size, False)

        return KL_loss+BCE


def mini_batch():
    for i in range(0, len(train_data), mb_size):
        yield train_data[i:i+mb_size]


encoder = cn.ConvoNet([1, 1, in_shape], cn.NoLoss())

encoder.add(cn.Dense(in_shape, hid_shape, cn.LeakyReLu()))
encoder.add(cn.Dense(hid_shape, lat_shape*2, cn.LeakyReLu(1)))

decoder = cn.ConvoNet(
    [1, 1, lat_shape], cn.BinaryCrossEntropyLoss(average=False))

decoder.add(cn.Dense(lat_shape, hid_shape, cn.LeakyReLu()))
decoder.add(cn.Dense(hid_shape, out_shape, cn.Sigmoid()))


model = VAE(encoder, decoder)

for e in range(epoch):
    training_data = mini_batch()
    print('epoch:', e)
    total_loss = 0
    bce = 0
    for i, data in enumerate(training_data):
        loss = model.train(data)
        total_loss += loss
    print('Total loss: ', total_loss/train_data.shape[0])


def plot(axes, image):
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(image[i*axes.shape[0]+j].reshape(28, 28))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])


fig, axes = plt.subplots(6, 6)

randoms = np.random.randn(36, lat_shape)
generated = decoder.predict(randoms)

plot(axes, generated)
plt.show()
