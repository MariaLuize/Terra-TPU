import tensorflow as tf

opticalBands   = ['green','red','nir','swir1']
opticalIndices = ['NDVI','MNDWI']
BANDS          = opticalBands + opticalIndices

RESPONSE = 'supervised's
FEATURES = BANDS + [RESPONSE]

KERNEL_SIZE  = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

def parse_tfrecord(example_proto):
  """The parsing function.
  Read a serialized example into the structure defined by FEATURES_DICT.
  Args:
    example_proto: a serialized Example.
  Returns: 
    A dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, FEATURES_DICT)

def to_tuple(inputs):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns: 
    A dtuple of (inputs, outputs).
  """
  inputsList = [inputs.get(key) for key in FEATURES]
  stacked = tf.stack(inputsList, axis=0)
  # Convert from CHW to HWC
  stacked = tf.transpose(stacked, [1, 2, 0])
  bandas_data = stacked[:,:,:len(BANDS)]/255.0
  label_data = stacked[:,:,len(BANDS):]
  return tf.cast(bands_data, tf.float32), tf.cast(label_data, tf.float32) 


def get_dataset(bucket_path, batch_size=64, is_training=True):
  """Function to read, parse and format to tuple a set of input tfrecord files.
  Get all the files matching the pattern, parse and convert to tuple.
  Args:
    pattern: A file pattern to match in a Cloud Storage bucket.
  Returns: 
    A tf.data.Dataset
  """
  glob = tf.io.gfile.glob(os.path.join(bucket_path, "*.tfrecord.gz"))
  if not glob:
        print(f"No files in {pattern}")
  dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.map(to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
  if is_training:
        dataset = dataset.shuffle(1000).repeat()
  
  # drop_remainder=True é recomendado para TPU para manter shapes estáticos
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset