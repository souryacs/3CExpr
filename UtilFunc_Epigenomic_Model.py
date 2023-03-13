##===================
## various utility functions related to training and testing the model
##===================

import tensorflow as tf

## loss function using true and predicted labels
def poisson_loss(y_true, mu_pred):
    nll = tf.reduce_mean(tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
    return nll

def poisson_loss_individual(y_true, mu_pred):
    nll = tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred)
    return nll

## read input TFRecord file
def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

## read the TFRecord according to the data prototype
def parse_proto(example_protos):
    features = {
        'last_batch': tf.io.FixedLenFeature([1], tf.int64),
        'adj': tf.io.FixedLenFeature([], tf.string),
        #'adj_real': tf.io.FixedLenFeature([], tf.string),
        'tss_idx': tf.io.FixedLenFeature([], tf.string),
        'X_1d': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'bin_idx': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.io.parse_example(example_protos, features=features)

    last_batch = parsed_features['last_batch']

    adj = tf.io.decode_raw(parsed_features['adj'], tf.float16)
    adj = tf.cast(adj, tf.float32)

    #adj_real = tf.io.decode_raw(parsed_features['adj_real'], tf.float16)
    #adj_real = tf.cast(adj_real, tf.float32)

    tss_idx = tf.io.decode_raw(parsed_features['tss_idx'], tf.float16)
    tss_idx = tf.cast(tss_idx, tf.float32)

    X_epi = tf.io.decode_raw(parsed_features['X_1d'], tf.float16)
    X_epi = tf.cast(X_epi, tf.float32)

    Y = tf.io.decode_raw(parsed_features['Y'], tf.float16)
    Y = tf.cast(Y, tf.float32)

    bin_idx = tf.io.decode_raw(parsed_features['bin_idx'], tf.int64)
    bin_idx = tf.cast(bin_idx, tf.int64)

    return {'last_batch': last_batch, 'X_epi': X_epi, 'Y': Y, 'adj': adj, 'tss_idx': tss_idx, 'bin_idx': bin_idx}


## read one particular batch
def read_tf_record_1shot_train(iterator, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack):
    try:
        ## compatible with tensorflow 1
        if 0:
            next_datum = iterator.get_next()
        ## compatible with tensorflow 2
        if 1:
            next_datum = next(iterator)
        data_exist = True

    # sourya - modify the error handler
    # except tf.errors.OutOfRangeError:
    except Exception as e:
        print('\n\n !!!!! Data read TF record - error occured ')
        data_exist = False
    
    if data_exist:

        ## sourya - here we are initializing same parameters
        ## to do: automate initializing these parameters / transfer the parameters 
        # T = 400 #num_slide_window_loop_bin       # number of 5kb bins inside middle 2Mb region 
        # b = 50  #ratio_loopbin_epibin       # number of 100bp bins inside 5Kb region
        # F = 6  #num_epitrack #3      # number of Epigenomic tracks used in model

        X_epi = next_datum['X_epi']
        batch_size = tf.shape(X_epi)[0]
        # X_epi = tf.reshape(X_epi, [batch_size, 3*T*b, F])
        X_epi = tf.reshape(X_epi, [batch_size, num_span_epi_bin, num_epitrack])
        
        adj = next_datum['adj']
        # adj = tf.reshape(adj, [batch_size, 3*T, 3*T])
        adj = tf.reshape(adj, [batch_size, num_span_loop_bin, num_span_loop_bin])

        if 0:
            last_batch = next_datum['last_batch']
        
        tss_idx = next_datum['tss_idx']
        # tss_idx = tf.reshape(tss_idx, [3*T])
        tss_idx = tf.reshape(tss_idx, [num_span_loop_bin])
        
        ## idx: middle region (sliding window)
        # idx = tf.range(T, 2*T)
        idx = tf.range(num_slide_window_loop_bin, 2*num_slide_window_loop_bin)

        Y = next_datum['Y']
        if 0:
            # Y = tf.reshape(Y, [batch_size, 3*T, b])
            Y = tf.reshape(Y, [batch_size, num_span_loop_bin, ratio_loopbin_epibin])
            Y = tf.reduce_sum(Y, axis=2)
        
        # Y = tf.reshape(Y, [batch_size, 3*T])
        Y = tf.reshape(Y, [batch_size, num_span_loop_bin])

    else:
        X_epi = 0
        Y = 0
        adj = 0
        tss_idx = 0
        idx = 0
    return data_exist, X_epi, Y, adj, idx, tss_idx


## read one dataset, according to the specified batch size
def dataset_iterator(file_name, batch_size):
    dataset = tf.data.Dataset.list_files(file_name)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_proto)
    ## compatible with tensorflow 1
    if 0:
        iterator = dataset.make_one_shot_iterator()
    ## compatible with tensorflow 2
    if 1:
        iterator = iter(dataset)
    return iterator


def set_axis_style(ax, labels, positions_tick):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xticks(positions_tick)
    ax.set_xticklabels(labels, fontsize=20)

def add_label(violin, labels, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

