import tensorlayer as tl
import tensorflow as tf

def get_train_data(img_hr_size, img_lr_size, batch_size, path_folder_with_images, shuffle_buffer_size=128):

    # List file names
    train_hr_img_list = sorted(tl.files.load_file_list(path=path_folder_with_images,
                                                       regx='.*.png',
                                                       printable=True))

    ## Pre-load train set
    train_hr_imgs = tl.vis.read_images(train_hr_img_list,
                                       path=path_folder_with_images,
                                       n_threads=32)


    def generator_train():
        for img in train_hr_imgs:
            yield img

    # Augmentation
    def _map_fn_train(img):

        # Crop Image (256, 256, 3)
        hr_patch = tf.image.random_crop(img, [img_hr_size, img_hr_size, 3])

        # Normalization [-1, 1]
        hr_patch = hr_patch / (255. / 2.) - 1.

        # Make Small Dataset (64, 64)
        lr_patch = tf.image.resize(hr_patch, size=[img_lr_size, img_lr_size])
        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())

    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2) # Next Iteration Cache
    train_ds = train_ds.batch(batch_size)
    return train_ds
