import numpy as np
import torch.nn as nn
from dragon.search_space.cells import Brick

class AVGPooling2D(Brick):
    def __init__(self, input_shape, pool,  stride=None):
        super(AVGPooling2D, self).__init__(input_shape)
        c, h, w = input_shape
        self.pool_size=(pool,pool)
        if stride is None:
            stride = pool
        self.stride = stride
        self.pooling = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.stride)

    def forward(self, X):
        if (self.pool_size[0]>X.shape[-2]) or (self.pool_size[1] > X.shape[-1]):
            X = self.pad(X)
        X = self.pooling(X)
        return X
    def pad(self,X):
        diff0 = self.pool_size[0] - X.shape[-2]
        sign0 = diff0 / np.abs(diff0) if diff0!=0 else 1
        diff1 = self.pool_size[1] - X.shape[-1]
        sign1 = diff1 / np.abs(diff1) if diff1!=0 else 1
        pad = (int(sign1 * np.ceil(np.abs(diff1)/2)), int(sign1 * np.floor(np.abs(diff1))/2),
               int(sign0 * np.ceil(np.abs(diff0)/2)), int(sign0 * np.floor(np.abs(diff0))/2))
        return nn.functional.pad(X, pad)
    
    def modify_operation(self, input_shape):
        pass

class MaxPooling2D(Brick):
    def __init__(self, input_shape, pool,  stride=None):
        super(MaxPooling2D, self).__init__(input_shape)
        c, h, w = input_shape
        self.pool_size = (pool,pool)
        if stride is None:
            stride = pool
        self.stride = stride
        self.pooling = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride)

    def forward(self, X):
        if (self.pool_size[0]>X.shape[-2]) or (self.pool_size[1] > X.shape[-1]):
            X = self.pad(X)
        X = self.pooling(X)
        return X
    
    def pad(self,X):
        diff0 = self.pool_size[0] - X.shape[-2]
        sign0 = diff0 / np.abs(diff0) if diff0!=0 else 1
        diff1 = self.pool_size[1] - X.shape[-1]
        sign1 = diff1 / np.abs(diff1) if diff1!=0 else 1
        pad = (int(sign1 * np.ceil(np.abs(diff1)/2)), int(sign1 * np.floor(np.abs(diff1))/2),
               int(sign0 * np.ceil(np.abs(diff0)/2)), int(sign0 * np.floor(np.abs(diff0))/2))
        return nn.functional.pad(X, pad)
    
    def modify_operation(self, input_shape):
        pass


class AVGPooling3D(nn.Module):
    def __init__(self, input_shape, pool, p2=None, p3=None, s1=None, s2=None, s3=None):
        super(AVGPooling3D, self).__init__()
        if p2 is None:
            p2 = pool
        if p3 is None:
            p3 = p2
        self.pool_size = (pool, p2, p3)
        if s1 is None:
            s1 = pool
        if s2 is None:
            s2 = p2
        if s3 is None:
            s3 = p3
        self.stride = (s1, s2, s3)
        self.pooling = nn.AvgPool3d(kernel_size=self.pool_size, stride=self.stride)

    def forward(self, X):
        if (self.pool_size[0]>X.shape[-3]) or (self.pool_size[1] > X.shape[-2]) or (self.pool_size[2] > X.shape[-1]):
            X = self.pad(X)
        X = self.pooling(X)
        return X
    def pad(self,X):
        diff0 = self.pool_size[0] - X.shape[-3]
        if diff0<=0:
            diff0 = 0
        diff1 = self.pool_size[1] - X.shape[-2]
        if diff1<=0:
            diff1 = 0
        diff2 = self.pool_size[2] - X.shape[-1]
        if diff2<=0:
            diff2 = 0
        pad = (int(np.ceil(np.abs(diff2)/2)), int(np.floor(np.abs(diff2))/2),
                int(np.ceil(np.abs(diff1)/2)), int(np.floor(np.abs(diff1))/2),
                int(np.ceil(np.abs(diff0)/2)), int(np.floor(np.abs(diff0))/2))
        return nn.functional.pad(X, pad)
    
    def modify_operation(self, input_shape):
        pass

class MaxPooling3D(nn.Module):
    def __init__(self, input_shape, pool, p2=None, p3=None, s1=None, s2=None, s3=None):
        super(MaxPooling3D, self).__init__()
        if p2 is None:
            p2 = pool
        if p3 is None:
            p3 = p2
        self.pool_size = (pool, p2, p3)
        if s1 is None:
            s1 = pool
        if s2 is None:
            s2 = p2
        if s3 is None:
            s3 = p3
        self.stride = (s1, s2, s3)
        self.pooling = nn.MaxPool3d(kernel_size=self.pool_size, stride=self.stride)

    def forward(self, X):
        if (self.pool_size[0]>X.shape[-3]) or (self.pool_size[1] > X.shape[-2]) or (self.pool_size[2] > X.shape[-1]):
            X = self.pad(X)
        X = self.pooling(X)
        return X
    def pad(self,X):
        diff0 = self.pool_size[0] - X.shape[-3]
        if diff0<=0:
            diff0 = 0
        diff1 = self.pool_size[1] - X.shape[-2]
        if diff1<=0:
            diff1 = 0
        diff2 = self.pool_size[2] - X.shape[-1]
        if diff2<=0:
            diff2 = 0
        pad = (int(np.ceil(np.abs(diff2)/2)), int(np.floor(np.abs(diff2))/2),
                int(np.ceil(np.abs(diff1)/2)), int(np.floor(np.abs(diff1))/2),
                int(np.ceil(np.abs(diff0)/2)), int(np.floor(np.abs(diff0))/2))
        return nn.functional.pad(X, pad)
    
    def modify_operation(self, input_shape):
        pass
    
    


class AVGPooling1D(Brick):
    def __init__(self, input_shape, pool_size):
        super(AVGPooling1D, self).__init__(input_shape)
        if len(input_shape) > 1:
            c, l = input_shape
        else:
            c = input_shape[0]
        self.pool_size = pool_size
        self.pooling = nn.AvgPool1d(pool_size)

    def forward(self, X):
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(1)
        if self.pool_size>X.shape[-1]:
            X = self.pad(X)
        X = self.pooling(X)
        if len(init_shape) < 3:
            X = X.squeeze(1)
        return X
    
    def modify_operation(self, input_shape):
        pass

    def pad(self,X): 
        diff = self.pool_size - X.shape[-1]
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        return nn.functional.pad(X, pad)


class MaxPooling1D(Brick):
    def __init__(self, input_shape, pool_size):
        super(MaxPooling1D, self).__init__(input_shape)
        self.pool_size=pool_size
        self.pooling = nn.MaxPool1d(pool_size)

    def forward(self, X):
        init_shape = X.shape
        if len(init_shape) < 3:
            X = X.unsqueeze(1)
        if self.pool_size>X.shape[-1]:
            X = self.pad(X)
        X = self.pooling(X)
        if len(init_shape) < 3:
            X = X.squeeze(1)
        return X
    
    def modify_operation(self, input_shape):
        pass

    def pad(self,X): 
        diff = self.pool_size - X.shape[-1]
        sign = diff / np.abs(diff) if diff !=0 else 1
        pad = (int(sign * np.ceil(np.abs(diff)/2)), int(sign * np.floor(np.abs(diff))/2))
        return nn.functional.pad(X, pad)
