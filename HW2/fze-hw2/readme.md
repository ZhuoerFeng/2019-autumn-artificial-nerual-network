说明:
目前代码为GPU上下载下来的文件，时间戳保持在ddl前。
``./mlp/model.py``中，是去掉bn层的代码

```python
# TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer
            dense1 = tf.layers.dense(inputs=self.x_, units=256, kernel_initializer= tf.truncated_normal_initializer(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
#             bn = batch_normalization_layer(dense1, is_train)
            # Your Relu Layer
            relu = tf.nn.relu(dense1)
            # Your Dropout Layer: use dropout_layer function
            drop = dropout_layer(relu, FLAGS.drop_rate, is_train)
            # Your Linear Layer
            logits = tf.layers.dense(drop, units = 10, kernel_initializer = tf.truncated_normal_initializer(stddev=0.1))
            # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

```

如果加上bn层，则为

```python
# TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer
            dense1 = tf.layers.dense(inputs=self.x_, units=256, kernel_initializer= tf.truncated_normal_initializer(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
            bn = batch_normalization_layer(dense1, is_train)
            # Your Relu Layer
            relu = tf.nn.relu(bn)
            # Your Dropout Layer: use dropout_layer function
            drop = dropout_layer(relu, FLAGS.drop_rate, is_train)
            # Your Linear Layer
            logits = tf.layers.dense(drop, units = 10, kernel_initializer = tf.truncated_normal_initializer(stddev=0.1))
            # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

```



``./cnn/model.py``中，亦是去掉BN的代码，为

```python
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Conv Layer ????
            conv1 = tf.layers.conv2d(self.x_, 40, 8, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
#             bn1 = batch_normalization_layer(conv1, is_train)
            # Your Relu Layer
            relu1 = tf.nn.relu(conv1)
            # Your Dropout Layer: use dropout_layer function
            drop1 = dropout_layer(relu1, FLAGS.drop_rate, is_train)
            # Your MaxPool
            pool1 = tf.layers.max_pooling2d(drop1, 2, 2)
            # Your Conv Layer ????
            conv2 = tf.layers.conv2d(pool1, 40, 8, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
#             bn2 = batch_normalization_layer(conv2, is_train)
            # Your Relu Layer
            relu2 = tf.nn.relu(conv2)
            # Your Dropout Layer: use dropout_layer function
            drop2 = dropout_layer(relu2, FLAGS.drop_rate, is_train)
            # Your MaxPool
            pool2 = tf.layers.max_pooling2d(drop2, 2, 2)
            # Your Linear Layer
            logits = tf.layers.dense(tf.reshape(pool2, [-1, 160]), 10,
kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
```

加上BN层后，应该是

```python
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Conv Layer ????
            conv1 = tf.layers.conv2d(self.x_, 40, 8, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
            bn1 = batch_normalization_layer(conv1, is_train)
            # Your Relu Layer
            relu1 = tf.nn.relu(bn1)
            # Your Dropout Layer: use dropout_layer function
            drop1 = dropout_layer(relu1, FLAGS.drop_rate, is_train)
            # Your MaxPool
            pool1 = tf.layers.max_pooling2d(drop1, 2, 2)
            # Your Conv Layer ????
            conv2 = tf.layers.conv2d(pool1, 40, 8, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            # Your BN Layer: use batch_normalization_layer function
            bn2 = batch_normalization_layer(conv2, is_train)
            # Your Relu Layer
            relu2 = tf.nn.relu(bn2)
            # Your Dropout Layer: use dropout_layer function
            drop2 = dropout_layer(relu2, FLAGS.drop_rate, is_train)
            # Your MaxPool
            pool2 = tf.layers.max_pooling2d(drop2, 2, 2)
            # Your Linear Layer
            logits = tf.layers.dense(tf.reshape(pool2, [-1, 160]), 10,
kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
```

两者的main.py的75行前后均有用于输出数据的变量添加，对本身运行逻辑并没有过多修改。

``./mlp/data``为mlp的结果数据

``./cnn/real``为cnn的结果数据

