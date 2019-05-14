import tensorflow as tf

c = []
for i, d in enumerate(['/gpu:0', '/gpu:1']):
    with tf.device(d):
        a = tf.get_variable(f"a_{i}", [2, 3], initializer=tf.random_uniform_initializer(-1, 1))
        b = tf.get_variable(f"b_{i}", [3, 2], initializer=tf.random_uniform_initializer(-1, 1))
        c.append(tf.matmul(a, b))

with tf.device('/cpu:0'):
    sum = tf.add_n(c)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(sum))