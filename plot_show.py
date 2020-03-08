import matplotlib.pyplot as plt
import tensorflow as tf

def show_plot_results(train_ds, pre_generator, gan_generator):
  for (lr, hr) in train_ds.take(1):

    pre_sr = pre_generator(lr)
    gan_sr = gan_generator(lr)

    lr = lr[0,:,:,:]
    pre_sr = pre_sr[0,:,:,:]
    gan_sr = gan_sr[0,:,:,:]

    lr = lr[:, :] * 127.5 + 127.5
    lr = tf.dtypes.cast(lr, tf.int32)

    pre_sr = pre_sr[:, :] * 127.5 + 127.5
    pre_sr = tf.dtypes.cast(pre_sr, tf.int32)

    gan_sr = gan_sr[:, :] * 127.5 + 127.5
    gan_sr = tf.dtypes.cast(gan_sr, tf.int32)


    hr = hr[0, :, :, :]
    hr = hr[:, :] * 127.5 + 127.5
    hr = tf.dtypes.cast(hr, tf.int32)

    psnr_pre = tf.image.psnr(pre_sr, hr, max_val=255)
    psnr_gan = tf.image.psnr(gan_sr, hr, max_val=255)

    plt.figure(figsize=(20, 20))

    images = [lr, hr, pre_sr, gan_sr]
    titles = ['LR', 'Original', f'SR (PRE) RSNR:{psnr_pre}', f'SR (GAN) - RSNR:{psnr_gan}']
    positions = [1, 2, 3, 4]


    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
