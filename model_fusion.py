import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_vgg16_branch(input_shape=(224,224,3)):
    base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    vgg_features = layers.Dense(40, activation='relu')(x)  # mRMR selection to 40 dims
    return base_model.input, vgg_features

def build_fusion_model(clinical_dim=42, radiomics_dim=76, img_shape=(224,224,3)):
    clinical_input = Input(shape=(clinical_dim,), name='clinical_input')
    radiomics_input = Input(shape=(radiomics_dim,), name='radiomics_input')
    img_input, deep_features = build_vgg16_branch(img_shape)

    concat = layers.Concatenate()([clinical_input, radiomics_input, deep_features])
    attention_logits = layers.Dense(3, activation='softmax', name='modality_attention')(concat)
    split_feats = tf.split(concat, [clinical_dim, radiomics_dim, 40], axis=1)
    weighted_feats = [layers.Multiply()([split_feats[i], attention_logits[:, i:i+1]]) for i in range(3)]
    fused = layers.Concatenate()(weighted_feats)
    x = layers.Dense(128, activation='relu')(fused)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(4, activation='softmax', name='output')(x)
    model = Model(inputs=[clinical_input, radiomics_input, img_input], outputs=output)
    return model