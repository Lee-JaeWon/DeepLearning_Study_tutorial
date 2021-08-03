{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5c4696c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision.datasets as dsets\n",
    "import torchvision.transforms as transforms\n",
    "\n",
    "import torch.nn.init"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "740e62d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Setting up a new session...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "''"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import visdom\n",
    "\n",
    "vis = visdom.Visdom()\n",
    "vis.close(env=\"main\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cc07eda2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss_tracker(loss_plot, loss_value, num):\n",
    "    '''num, loss_value, are Tensor'''\n",
    "    vis.line(X=num,\n",
    "             Y=loss_value,\n",
    "             win = loss_plot,\n",
    "             update='append'\n",
    "             )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "99107816",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "다음 기기로 학습합니다: cpu\n"
     ]
    }
   ],
   "source": [
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "print(\"다음 기기로 학습합니다:\", device)\n",
    "\n",
    "torch.manual_seed(777)\n",
    "if device =='cuda':\n",
    "    torch.cuda.manual_seed_all(777)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c296e87d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#parameters\n",
    "learning_rate = 0.001\n",
    "training_epochs = 15\n",
    "batch_size = 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2c356bdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\LeeJaeWon\\anaconda3\\envs\\DeepLearningStudy\\lib\\site-packages\\torchvision\\datasets\\mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\\torch\\csrc\\utils\\tensor_numpy.cpp:180.)\n",
      "  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)\n"
     ]
    }
   ],
   "source": [
    "#MNIST dataset\n",
    "\n",
    "mnist_train = dsets.MNIST(root='E:\\DeepLearningStudy',\n",
    "                          train=True,\n",
    "                          transform=transforms.ToTensor(),\n",
    "                          download=True)\n",
    "\n",
    "mnist_test = dsets.MNIST(root='E:\\DeepLearningStudy',\n",
    "                         train=False,\n",
    "                         transform=transforms.ToTensor(),\n",
    "                         download=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a093f757",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_loader = torch.utils.data.DataLoader(dataset=mnist_train,\n",
    "                                          batch_size = batch_size,\n",
    "                                          shuffle =True,\n",
    "                                          drop_last=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d64e664e",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CNN(nn.Module):\n",
    "    \n",
    "    def __init__(self):\n",
    "        super(CNN, self).__init__()\n",
    "        self.layer1 = nn.Sequential(\n",
    "            nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2)\n",
    "        )\n",
    "        \n",
    "        self.layer2 = nn.Sequential(\n",
    "            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2)\n",
    "        )\n",
    "        \n",
    "        self.layer3 = nn.Sequential(\n",
    "            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2)\n",
    "        )\n",
    "        \n",
    "        self.fc1 = nn.Linear(3*3*128, 625)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.fc2 = nn.Linear(625, 10, bias =True)\n",
    "        torch.nn.init.xavier_uniform_(self.fc1.weight)\n",
    "        torch.nn.init.xavier_uniform_(self.fc2.weight)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        out = self.layer1(x)\n",
    "        out = self.layer2(out)\n",
    "        out = self.layer3(out)\n",
    "        \n",
    "        out = out.view(out.size(0), -1)\n",
    "        out = self.fc1(out)\n",
    "        out = self.relu(out)\n",
    "        out = self.fc2(out)\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "44f50ec3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 10])\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\LeeJaeWon\\anaconda3\\envs\\DeepLearningStudy\\lib\\site-packages\\torch\\nn\\functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  ..\\c10/core/TensorImpl.h:1156.)\n",
      "  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)\n"
     ]
    }
   ],
   "source": [
    "model = CNN().to(device)\n",
    "\n",
    "value = (torch.Tensor(1,1,28,28)).to(device)\n",
    "print( (model(value)).shape )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "eecc4c8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss().to(device)\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "16c968af",
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9880b17e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch:1] cost = 0.1273089200258255\n",
      "[Epoch:2] cost = 0.04218006506562233\n",
      "[Epoch:3] cost = 0.029042549431324005\n",
      "[Epoch:4] cost = 0.02246006578207016\n",
      "[Epoch:5] cost = 0.018458593636751175\n",
      "[Epoch:6] cost = 0.014476736076176167\n",
      "[Epoch:7] cost = 0.013362587429583073\n",
      "[Epoch:8] cost = 0.010848738253116608\n",
      "[Epoch:9] cost = 0.00970136933028698\n",
      "[Epoch:10] cost = 0.00856876838952303\n",
      "[Epoch:11] cost = 0.008588884025812149\n",
      "[Epoch:12] cost = 0.008137150667607784\n",
      "[Epoch:13] cost = 0.007051507476717234\n",
      "[Epoch:14] cost = 0.007606089115142822\n",
      "[Epoch:15] cost = 0.00602350290864706\n",
      "Learning Finished!\n"
     ]
    }
   ],
   "source": [
    "#training\n",
    "total_batch = len(data_loader)\n",
    "\n",
    "for epoch in range(training_epochs):\n",
    "    avg_cost = 0\n",
    "    \n",
    "    for X, Y in data_loader:\n",
    "        X = X.to(device)\n",
    "        Y = Y.to(device)\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        hypothesis = model(X)\n",
    "        \n",
    "        cost = criterion(hypothesis, Y)\n",
    "        cost.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        avg_cost += cost / total_batch\n",
    "    \n",
    "    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))\n",
    "    loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))\n",
    "print('Learning Finished!')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "409f9f17",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\LeeJaeWon\\anaconda3\\envs\\DeepLearningStudy\\lib\\site-packages\\torchvision\\datasets\\mnist.py:67: UserWarning: test_data has been renamed data\n",
      "  warnings.warn(\"test_data has been renamed data\")\n",
      "C:\\Users\\LeeJaeWon\\anaconda3\\envs\\DeepLearningStudy\\lib\\site-packages\\torchvision\\datasets\\mnist.py:57: UserWarning: test_labels has been renamed targets\n",
      "  warnings.warn(\"test_labels has been renamed targets\")\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.983299970626831\n"
     ]
    }
   ],
   "source": [
    "with torch.no_grad():\n",
    "    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)\n",
    "    Y_test = mnist_test.test_labels.to(device)\n",
    "    \n",
    "    prediction = model(X_test)\n",
    "    correct_prediction = torch.argmax(prediction, 1) == Y_test\n",
    "    accuracy = correct_prediction.float().mean() \n",
    "    print('Accuracy:', accuracy.item())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
