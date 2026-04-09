import tensorflow as tf

# FEATURES_DICT = {
#     'green': tf.io.FixedLenFeature([256, 256], tf.float32),
#     'red': tf.io.FixedLenFeature([256, 256], tf.float32),
#     'nir': tf.io.FixedLenFeature([256, 256], tf.float32),
#     'swir1': tf.io.FixedLenFeature([256, 256], tf.float32),
#     'NDVI': tf.io.FixedLenFeature([256, 256], tf.float32),
#     'MNDWI': tf.io.FixedLenFeature([256, 256], tf.float32),
#     'supervised': tf.io.FixedLenFeature([256, 256], tf.float32),
# }

opticalBands   = ['green','red','nir','swir1']
opticalIndices = ['NDVI','MNDWI']
BANDS          = opticalBands + opticalIndices

RESPONSE = 'supervised'
FEATURES = BANDS + [RESPONSE]

KERNEL_SIZE  = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

def parse_tfrecord(example_proto):
    return tf.io.parse_single_example(example_proto, FEATURES_DICT)

def to_tuple(inputs):
    input_list = [inputs.get(key) for key in FEATURES]
    stacked = tf.stack(input_list, axis=-1)
    label = tf.expand_dims(inputs.get('supervised'), axis=-1)
    return stacked, label

def get_dataset(bucket_path, batch_size=64):
    glob    = tf.io.gfile.glob(bucket_path + "/*.tfrecord.gz")
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset