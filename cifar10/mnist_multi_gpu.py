
import tensorflow as tf



def core_model(input_img, num_classes=10):
    """
        A simple model to perform classification on 28x28 grayscale images in MNIST style.

        Args:
        input_img:  A floating point tensor with a shape that is reshapable to batchsizex28x28. It
            represents the inputs to the model
        num_classes:  The number of classes
    """
    net = tf.reshape(input_img, [-1, 28, 28, 1])
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu,
                           name="conv2d_1")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu,
                           name="conv2d_2")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.reshape(net, [-1, 7 * 7 * 64])
    net = tf.layers.dense(inputs=net, units=1024, name="dense_1", activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=net, units=num_classes, name="dense_2")
    return logits


def training_model(input_fn):
    inputs = input_fn()
    image = inputs[0]
    label = tf.cast(inputs[1], tf.int32)
    logits = core_model(image)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
    return tf.reduce_mean(loss)


def training_dataset(epochs=5, batch_size=128):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets("data")
    all_data_points = mnist_data.train.next_batch(60000)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(all_data_points)
    dataset = dataset.repeat(epochs).shuffle(10000).batch(batch_size)
    return dataset


def do_training(update_op, loss):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            step = 0
            while True:
                _, loss_value = sess.run((update_op, loss))
                if step % 100 == 0:
                    print('Step {} with loss {}'.format(step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final loss: {}'.format(loss_value))


def serial_training(model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()

    loss = model_fn(lambda: iterator.get_next())
    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    global_step = tf.train.get_or_create_global_step()
    update_op = optimizer.minimize(loss, global_step=global_step)

    do_training(update_op, loss)


tf.reset_default_graph()
serial_training(training_model, training_dataset(epochs=2))