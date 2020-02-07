from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from datetime import datetime

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
# model = Network()
# model.add(Linear('fc1', 784, 256, 0.01))
# model.add(Relu('fc2'))
# model.add(Linear('fc3', 256, 10, 0.01))
# model.add(Relu('fc4'))

# # loss = EuclideanLoss(name='loss')
# loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.
'''
config = {
    'learning_rate': 0.03,
    'weight_decay': 0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}
'''

def no_hidden_layer():
    filename = "./out/no_hidden_layer.txt"
    model = Network()
    model.add(Linear('fc1', 784, 10, 0.001))
    loss = EuclideanLoss(name='loss')

    config = {
        'learning_rate': 0.00001,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss

def double_layer_with_sigmoid():
    filename = "./out/double_layer_with_sigmoid_softmax.txt"
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Sigmoid('sg1'))
    model.add(Linear('fc2', 256, 10, 0.001))
    model.add(Sigmoid('sg2'))
    loss = SoftmaxCrossEntropyLoss(name='loss')

    config = {
        'learning_rate': 0.03,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss

def double_layer_with_relu():
    filename = "./parameters/double_layer_with_relu_momentum01.txt"
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 256, 10, 0.001))
    model.add(Relu('rl2'))
    loss = EuclideanLoss(name='loss')

    config = {
        'learning_rate': 0.03,
        'weight_decay': 0,
        'momentum': 0.1,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss

def triple_layer_with_sigmoid():
    filename = "./out/triple_layer_with_sigmoid_softmax.txt"
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('sg1'))
    model.add(Linear('fc2', 256, 128, 0.01))
    model.add(Sigmoid('sg2'))
    model.add(Linear('fc3', 128, 10, 0.01))
    model.add(Sigmoid('sg3'))
    loss = EuclideanLoss(name='loss')

    config = {
        'learning_rate': 0.08,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss

def triple_layer_with_relu():
    filename = "./out/triple_layer_with_relu_more_nodes.txt"
    model = Network()
    model.add(Linear('fc1', 784, 512, 0.01)) # 128
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 512, 256, 0.01))
    model.add(Relu('rl2'))
    model.add(Linear('fc3', 256, 10, 0.01))
    model.add(Relu('rl3'))
    loss = EuclideanLoss(name='loss')

    config = {
        'learning_rate': 0.03, #0.03
        'weight_decay': 0, #0
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss

def double_layer_with_relu_softmax():
    filename = "./parameters/double_layer_with_relu_softmax.txt"
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.001))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 256, 10, 0.001))
    # model.add(Relu('rl2'))
    loss = SoftmaxCrossEntropyLoss(name='loss')

    config = {
        'learning_rate': 0.03,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss


def triple_layer_with_relu_softmax():
    filename = "./out/triple_layer_with_relu_softmax.txt"
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('rl1'))
    model.add(Linear('fc2', 256, 128, 0.01))
    model.add(Relu('rl2'))
    model.add(Linear('fc3', 128, 10, 0.01))
    model.add(Relu('rl3'))
    loss = SoftmaxCrossEntropyLoss(name='loss')

    config = {
        'learning_rate': 0.08,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 5
    }
    return model, config, filename, loss

record_for_acc = [ ]
record_for_loss = [ ]
record_for_test_acc = []
record_for_test_loss = []

model, config, filename, loss = triple_layer_with_relu()
time1 = datetime.now()
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    acc_record, loss_record = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    record_for_acc += acc_record
    record_for_loss += loss_record

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_acc, test_loss = test_net(model, loss, test_data, test_label, config['batch_size'])
        record_for_test_acc += test_acc 
        record_for_test_loss += test_loss
time2 = datetime.now()

with open(filename, 'w', encoding="utf-8") as f:
    f.write(str(record_for_acc))
    f.write("\n")
    f.write(str(record_for_loss))
    f.write("\n")
    f.write(str(record_for_test_acc))
    f.write("\n")
    f.write(str(record_for_test_loss))
    f.write("\n")
    f.write(str(time2 - time1))


