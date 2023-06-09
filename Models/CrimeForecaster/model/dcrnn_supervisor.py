from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml
from sklearn import metrics as metrics_sk
from sklearn.preprocessing import MinMaxScaler
import pickle

from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import cross_entropy, masked_mae_loss, masked_mse_loss, masked_rmse_loss

from model.dcrnn_model import DCRNNModel

import statistics

def sigmoid(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            # print("a :", array[i][j])
            array[i][j] = 1/(1 + np.exp(-array[i][j]))
            # print("b :", array[i][j])
    return array

class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data preparation
        self._data = utils.load_dataset(**self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=adj_mx, **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0
        # self._loss_fn = masked_rmse_loss(scaler, null_val)
        # self._loss_fn = masked_mae_loss(scaler, null_val)
        self._loss_fn = cross_entropy()
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mae': loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=1, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training ...')

        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                self._logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                                     [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_crossentropy: {:.4f}, val_cross_entropy: {:.4f} lr:{:.6f} {' \
                      ':.1f}s'.format(
                self._epoch, epochs, global_step, train_mae, val_mae, new_lr, (end_time - start_time))
            self._logger.info(message)
            # if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
            
            # self.evaluate(sess, self._epoch)  # commented by rittik

            ##############################################################################################################
            flag = self.evaluate(sess, self._epoch)
            if flag:
                break
            ##############################################################################################################

            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self, sess, epoch, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                return_output=True,
                                                training=False)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        y_preds = np.concatenate(y_preds, axis=0)
        y_truth = self._data['y_test'][:, :, :, :]
        y_pred = y_preds[:y_truth.shape[0], :, :, :]
        print("non zero elements before doing anything is {}".format(np.count_nonzero(y_pred)))
       
        #######################################################################################################################
        for i in range(77):
            y_truth_i = y_truth[:,:,i:i+1, :]
            y_pred_i = y_pred[:,:,i:i+1, :]

            y_truth_reshape_i = np.reshape(y_truth_i, (-1, 8))
            y_pred_reshape_i = np.reshape(y_pred_i, (-1, 8))
            y_pred_reshape_sigmoid_i = sigmoid(y_pred_reshape_i)
            ss = MinMaxScaler(feature_range=(0, 1))
            y_pred_reshape_sigmoid_i = ss.fit_transform(y_pred_reshape_sigmoid_i)
            
            # for regression
            mae = metrics_sk.mean_absolute_error(y_truth_reshape_i, y_pred_reshape_sigmoid_i)
            mse = metrics_sk.mean_squared_error(y_truth_reshape_i, y_pred_reshape_sigmoid_i)
            
            log_dir = "./result/"+str(epoch)+"/"
            #create folder for each epoch
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # save mse mae in txt file
            with open(log_dir + "mae_mse.txt", 'a') as f:
                f.write(f' {i} {mae} {mse}\n')

            # for classification
            # y_pred_reshape_sigmoid_i[y_pred_reshape_sigmoid_i >= 0.5] = 1
            # y_pred_reshape_sigmoid_i[y_pred_reshape_sigmoid_i < 0.5] = 0

            # macro_f1 = metrics_sk.f1_score(y_truth_reshape_i, y_pred_reshape_sigmoid_i, average = 'macro')
            # micro_f1 = metrics_sk.f1_score(y_truth_reshape_i, y_pred_reshape_sigmoid_i, average = 'micro')
            
            # #create folder for each epoch
            # if not os.path.exists(self._log_dir + "/" + str(epoch)):
            #     os.makedirs(self._log_dir + "/" + str(epoch))


            # # save f1 scores in txt file
            # with open(self._log_dir + "/" + str(epoch) +"/macro_micro.txt", 'a') as f:
            #     f.write(f' {i} {macro_f1} {micro_f1}\n')
                



            # make dir for each i
            if not os.path.exists(log_dir + str(i)):
                os.makedirs(log_dir  + str(i))
            
            # save y_pred_reshape_sigmoid_i and y_truth_reshape_i in pkl file
            with open(log_dir + str(i) + "/" + "y_truth.pkl", 'wb') as f:
                pickle.dump(y_truth_reshape_i, f)
            with open(log_dir + str(i) + "/" + "y_pred_true.pkl", 'wb') as f:
                pickle.dump(y_pred_reshape_sigmoid_i, f)

            #####################################################################################################################


        shape_vector = np.shape(y_truth)
        y_truth_reshape = np.reshape(y_truth, (-1, 8))
        y_pred_reshape = np.reshape(y_pred, (-1, 8))

        #cl_decay_steps for i in range(8):
        #     print(len(y_pred_reshape[:, i]))

        y_pred_reshape_sigmoid = sigmoid(y_pred_reshape)
        ss = MinMaxScaler(feature_range=(0, 1))
        y_pred_reshape_sigmoid = ss.fit_transform(y_pred_reshape_sigmoid)
        y_pred_reshape_sigmoid[y_pred_reshape_sigmoid >= 0.5] = 1
        y_pred_reshape_sigmoid[y_pred_reshape_sigmoid < 0.5] = 0


        # commented out by rittik

        # city = 'LA'
        # month = 8
        # print(np.shape(y_truth_reshape), np.shape(y_pred_reshape_sigmoid))
        # pickle.dump(y_truth_reshape, open('./result/%s/DCRNN/labels_testDCRNN%s%d.pkl'%(city,city,month), 'wb'))
        # pickle.dump(y_pred_reshape_sigmoid, open('./result/%s/DCRNN/predictDCRNN%s%d.pkl'%(city,city,month), 'wb')) 

        print("non-zero elements in prediction is {} and in truth is {} ".format(np.count_nonzero(
            y_pred_reshape_sigmoid), np.count_nonzero(y_truth_reshape)))
        
        ###########################################################################################################################
        with open('./result/non_zero_count.txt', 'a') as f:
            f.write(str(np.count_nonzero(y_truth_reshape))+" "+str(np.count_nonzero(y_pred_reshape_sigmoid))+'\n')
        
        if(np.count_nonzero(y_pred_reshape_sigmoid) == 0):
            print("Stopping as all zero")
            return True
        ###########################################################################################################################

        # commented out by rittik

        # macro_f1 = metrics_sk.f1_score(y_truth_reshape, y_pred_reshape_sigmoid, average = 'macro') 
        # micro_f1 = metrics_sk.f1_score(y_truth_reshape, y_pred_reshape_sigmoid, average = 'micro')

        # print("The average macro-F1 score is {}, average micro-F1 score is {}".format(macro_f1, micro_f1))
        # outputs = {
        #     'predictions': y_pred,
        #     'groundtruth': y_truth
        # }

        #return outputs 

        return False

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']
