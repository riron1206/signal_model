"""
2次元画像に見たてた信号データ用モデル
"""
import tensorflow as tf
import tensorflow.keras.layers as layers


def create_vgg_2d(input_shape=(80, 3), num_classes=3, activation="softmax"):
    """2次元データのVGGっぽいの"""
    inputs = layers.Input(input_shape)
    x = inputs
    for ch in [64, 128, 256]:
        x = layers.Conv1D(ch, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if ch != 256:
            x = layers.AveragePooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation=activation)(x)
    return tf.keras.models.Model(inputs, x)


def create_resnet_2d(input_shape=(80, 3), num_classes=3, activation="softmax"):
    """2次元データのresnet"""
    def residual_block(inputs, ch, strides):
        # main path
        x = layers.BatchNormalization()(inputs)
        x = layers.ReLU()(x)
        x = layers.Conv1D(ch, 3, strides=strides, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(ch, 3, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        # shortcut path
        if inputs.shape[-1] != ch or strides > 1:
            s = layers.Conv1D(ch, 3, strides=strides, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        else:
            s = inputs
        # add
        x = layers.Add()([x, s])
        return x

    def main():
        inputs = layers.Input(input_shape)
        x = layers.Conv1D(16, 3, padding="same")(inputs)
        for ch in [16, 32, 64]:
            for i in range(7):
                strides = 2 if i == 0 else 1
                if ch == 16:
                    strides = 1
                x = residual_block(x, ch, strides)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(num_classes, activation=activation)(x)

        return tf.keras.models.Model(inputs, x)

    return main()