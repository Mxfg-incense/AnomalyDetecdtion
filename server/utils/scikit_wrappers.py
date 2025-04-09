import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection

import utils.dataset_wrappers as dataset_wrappers
import losses
import networks
import joblib


class TimeSeriesEncoder(sklearn.base.BaseEstimator):
    """
    "Virtual" class to wrap an encoder of time series as a PyTorch module.
    
    All inheriting classes should implement the get_params and set_params
    methods, as in the recommendations of scikit-learn.

    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param encoder Encoder PyTorch module.
    @param params Dictionaries of the parameters of the encoder.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 encoder, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def load(self, prefix_file="combined"):
        """
        加载编码器。

        @param prefix_file 模型加载路径和前缀,模型将从'$(prefix_file)_$(architecture)_encoder.pth'加载
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def encode(self, X, batch_size=50):
        """
        输出编码器对输入数据的表示。

        @param X 测试集。
        @param batch_size 用于拆分测试数据的批次大小,以避免使用CUDA时内存溢出。
                         如果测试集包含不等长时间序列则忽略此参数。
        """
        # 检查给定的时间序列是否长度不等
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = dataset_wrappers.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    # 2024-07-31: 当整个序列都是nan时的快速修复
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu() if length != 0 else numpy.nan
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
            be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def fit(self, X, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Check if the given time series have unequal lengths       
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        
        train = train.cuda(self.gpu) if self.cuda else train

        train_torch_dataset = dataset_wrappers.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training  
        print('Training encoder...')
        i = 0  # 初始化步骤计数器
        epochs = 0  # 初始化 epoch 计数器

        while i < self.nb_steps:
            if verbose: 
                print(f'\nEpoch: {epochs + 1}')

            for batch_idx, batch in enumerate(train_generator):
                if self.cuda:
                    batch = batch.cuda(self.gpu)        
                    
                self.optimizer.zero_grad()

                if not varying:
                    loss = self.loss(batch, self.encoder, train, save_memory=save_memory)
                else:
                    loss = self.loss_varying(batch, self.encoder, train, save_memory=save_memory)   
                    
                loss.backward()
                self.optimizer.step()
                i += 1

                if verbose:
                    print(f'Step: {i}/{self.nb_steps}, Loss: {loss.item():.1f}, Batch: {batch_idx + 1}/{len(train_generator)}')

                if i >= self.nb_steps:
                    break

            # 更新 epoch 计数器
            epochs += 1

        return self.encoder  

class CausalCNNEncoder(TimeSeriesEncoder):
    def __init__(self, compared_length=50, nb_random_samples=10, 
                 negative_penalty=1, batch_size=1, nb_steps=20, 
                 lr=0.001, penalty=1, early_stopping=None, channels=10, 
                 depth=1, reduced_size=10, out_channels=10, kernel_size=4,
                in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoder, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels, kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self

class SVMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):

    def __init__(self, penalty=1):
        self.penalty = penalty
        self.classifier = sklearn.svm.SVC(
            C=1 / self.penalty if self.penalty is not None and self.penalty > 0 else numpy.inf,
            gamma='scale', max_iter=1000000
        )

    def fit(self, features, y):
        """
        使用预计算的特征训练分类器。使用RBF核的SVM分类器。

        @param features 训练集的计算特征。
        @param y 训练标签。
        """
        nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
        train_size = numpy.shape(features)[0]

        if train_size // nb_classes < 5 or train_size < 50 or self.penalty is not None:
            return self.classifier.fit(features, y)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(
                self.classifier, {
                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, numpy.inf],
                    'kernel': ['rbf'],
                    'degree': [3],
                    'gamma': ['scale'],
                    'coef0': [0],
                    'shrinking': [True],
                    'probability': [False],
                    'tol': [0.001],
                    'cache_size': [200],
                    'class_weight': [None],
                    'verbose': [False],
                    'max_iter': [10000000],
                    'decision_function_shape': ['ovr'],
                    'random_state': [None]
                },
                cv=5, n_jobs=5
            )
            if train_size <= 10000:
                grid_search.fit(features, y)
            else:
                # 如果训练集太大，随机抽样10000个样本进行训练
                split = sklearn.model_selection.train_test_split(
                    features, y,
                    train_size=10000, random_state=0, stratify=y
                )
                grid_search.fit(split[0], split[2])
            self.classifier = grid_search.best_estimator_
            return self.classifier

    def predict(self, features):
        """
        对给定的测试数据进行类别预测。

        @param features 测试集的特征。
        """
        return self.classifier.predict(features)

    def score(self, features, y):
        """
        在给定的测试数据上输出SVM分类器的准确率。

        @param features 测试集的特征。
        @param y 测试标签。
        """
        return self.classifier.score(features, y)

    def get_params(self, deep=True):
        return {
            'penalty': self.penalty
        }

    def set_params(self, penalty):
        self.__init__(penalty)
        return self

    def save(self, prefix_file):
        """
        保存SVM分类器。

        @param prefix_file 模型保存路径和前缀
        """
        joblib.dump(
            self.classifier,
            prefix_file + '_svm_classifier.pkl'
        )

    def load(self, prefix_file):
        """
        加载SVM分类器。

        @param prefix_file 模型加载路径和前缀
        """
        self.classifier = joblib.load(
            prefix_file + '_svm_classifier.pkl'
        )
