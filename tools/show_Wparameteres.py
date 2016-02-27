import cPickle
import numpy as np
import matplotlib.pyplot as plt

file = open("F:/model1.bin", 'rb')
params = cPickle.load(file)

W = params[4]
print W.shape

Wimg = W.transpose(0, 2, 3, 1)
print Wimg.shape


f, axs = plt.subplots(1, 20)

plt.gray()
for i in xrange(20):
    axs[i].imshow(Wimg[i, :, :, 2])

plt.show()
