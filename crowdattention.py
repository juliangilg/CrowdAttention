from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
import tensorflow as tf

class CrowdAttentionLayer(Layer):
    def __init__(self, K=0, R=0, **kwargs):
        super(CrowdAttentionLayer, self).__init__(**kwargs)
        self.R = R # Number of annotators.
        self.K = K # Number of classes.

    def scaled_dot_product_attention(self, q, k, v):
        # Compute the attention scores by employing the dot product between 
        # the DNN output and the labels from multiple annotators
        matmul_qk = tf.matmul(q, k, transpose_b=True) 
        attention_weights = (matmul_qk)

        Normalization = tf.math.reduce_sum(attention_weights, axis=-1, keepdims=True, name=None)
        output = tf.matmul(attention_weights, v)/(Normalization+1e-12) 
        

        return output, attention_weights

    def call(self, query_input, key_value_input):
        # Represent each label using the one-hot econding
        key_value_input = tf.cast(key_value_input, dtype=tf.int32)
        key_value_input = tf.one_hot(key_value_input-1, self.K)

        query_input = tf.expand_dims(query_input, axis=1)
        q = query_input
        k = key_value_input
        v = key_value_input

        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v)
        output = Reshape((self.K,))(attention_output)
        attention_weights = Reshape((self.R,))(attention_weights)
        return output, attention_weights


class CrowdAttentionLoss(Loss):
    def __init__(
        self,
        K: int = 0,
        R: int = 0,
    ) -> None:
        super().__init__()
        self.K = K
        self.R = R

    def call(self, y_true, y_pred):
      y_hat = y_pred[:,:self.K]
      y_tilde = y_pred[:,self.K:2*self.K]
      return self.cce(y_hat,y_tilde)

    def cce(self, y_true, y_pred):
      eps = 1e-9
      y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
      return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1))
