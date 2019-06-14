# Revival: Convert Minecraft to Real World
# CS 175: Project in Artificial Intelligence (Spring 2019)
#
# cyclegan.py
#

import csv
import os.path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from networks import Discriminator, CycleGenerator


LEARNING_RATE = 0.0002
# LEARNING_RATE = 0.0005
BETAS = (0.5, 0.999)
LAMBDA1 = 10
# LAMBDA2 = 0.5*LAMBDA1


class CycleGAN(object):
    @staticmethod
    def d_loss(d_net, real_image, fake_image):
        # return F.mse_loss(d_net(real_image), torch.tensor(1.0).to(device)) + \
        #        F.mse_loss(d_net(fake_image), torch.tensor(0.0).to(device))
        return torch.mean((d_net(real_image)-1)**2) + torch.mean(d_net(fake_image)**2)

    @staticmethod
    def g_loss(d_net, fake_image):
        # return F.mse_loss(d_net(fake_image), torch.tensor(1.0).to(device))
        return torch.mean((d_net(fake_image)-1)**2)

    @staticmethod
    def cycle_loss(real_image, reconstructed_image):
        # return F.l1_loss(reconstructed_image, real_image) * LAMBDA
        return torch.mean(torch.abs(reconstructed_image-real_image)) * LAMBDA1

    # @staticmethod
    # def identity_loss(g_net, real_image):
    #     return torch.mean(torch.abs(g_net(real_image)-real_image)) * LAMBDA2

    @staticmethod
    def tensor_to_numpy(tensor_image):
        if torch.cuda.is_available():
            tensor_image = tensor_image.cpu()
        image_numpy = tensor_image[0].float().detach().numpy()
        image_numpy = (image_numpy.transpose((1, 2, 0)) + 1) / 2 * 255
        return image_numpy.astype(np.uint8)

    @staticmethod
    def save_image(numpy_image, image_path):
        image = Image.fromarray(numpy_image)
        image.save(image_path)

    def __init__(self):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

        self.G_XtoY = CycleGenerator()
        self.G_YtoX = CycleGenerator()
        self.D_X = Discriminator()
        self.D_Y = Discriminator()

        if torch.cuda.is_available():
            self.G_XtoY.to(self.device)
            self.G_YtoX.to(self.device)
            self.D_X.to(self.device)
            self.D_Y.to(self.device)

    def load_model(self):
        G_XtoY_path = './checkpoints/G_XtoY.pkl'
        G_YtoX_path = './checkpoints/G_YtoX.pkl'
        D_X_path = './checkpoints/D_X.pkl'
        D_Y_path = './checkpoints/D_Y.pkl'

        self.G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
        self.G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
        self.D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
        self.D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

        if torch.cuda.is_available():
            self.G_XtoY.to(self.device)
            self.G_YtoX.to(self.device)
            self.D_X.to(self.device)
            self.D_Y.to(self.device)

    def save_model(self):
        G_XtoY_path = './checkpoints/G_XtoY.pkl'
        G_YtoX_path = './checkpoints/G_YtoX.pkl'
        D_X_path = './checkpoints/D_X.pkl'
        D_Y_path = './checkpoints/D_Y.pkl'
        torch.save(self.G_XtoY.state_dict(), G_XtoY_path)
        torch.save(self.G_YtoX.state_dict(), G_YtoX_path)
        torch.save(self.D_X.state_dict(), D_X_path)
        torch.save(self.D_Y.state_dict(), D_Y_path)

    def create_optimizer(self):
        self.g_optimizer = optim.Adam(list(self.G_XtoY.parameters())+list(self.G_YtoX.parameters()), LEARNING_RATE, BETAS)
        self.d_optimizer = optim.Adam(list(self.D_X.parameters())+list(self.D_Y.parameters()), LEARNING_RATE, BETAS)

    def load_data(self, data):
        self.image_real_X, _ = data['X']
        self.image_real_Y, _ = data['Y']
        if torch.cuda.is_available():
            self.image_real_X = self.image_real_X.to(self.device)
            self.image_real_Y = self.image_real_Y.to(self.device)

    def forward(self):
        self.image_fake_X = self.G_YtoX(self.image_real_Y)
        self.image_fake_Y = self.G_XtoY(self.image_real_X)
        self.image_reconstructed_X = self.G_YtoX(self.image_fake_Y)
        self.image_reconstructed_Y = self.G_XtoY(self.image_fake_X)

    def backward_d(self):
        self.D_X_Loss = CycleGAN.d_loss(self.D_X, self.image_real_X, self.image_fake_X)
        self.D_X_Loss.backward()
        self.D_Y_Loss = CycleGAN.d_loss(self.D_Y, self.image_real_Y, self.image_fake_Y)
        self.D_Y_Loss.backward()
        self.D_Loss = self.D_X_Loss + self.D_Y_Loss
        # self.D_Loss.backward()

    def backward_g(self):
        self.G_XtoY_Loss = CycleGAN.g_loss(self.D_Y, self.image_fake_Y)
        self.G_XtoY_Cycle_Loss = CycleGAN.cycle_loss(self.image_real_X, self.image_reconstructed_X)
        # self.G_XtoY_identity_Loss = CycleGAN.identity_loss(self.G_XtoY, self.image_real_Y)
        self.G_XtoYtoX_Loss = self.G_XtoY_Loss + self.G_XtoY_Cycle_Loss
        self.G_XtoYtoX_Loss.backward()

        self.G_YtoX_Loss = CycleGAN.g_loss(self.D_X, self.image_fake_X)
        self.G_YtoX_Cycle_Loss = CycleGAN.cycle_loss(self.image_real_Y, self.image_reconstructed_Y)
        # self.G_YtoX_identity_Loss = CycleGAN.identity_loss(self.G_YtoX, self.image_real_X)
        self.G_YtoXtoY_Loss = self.G_YtoX_Loss + self.G_YtoX_Cycle_Loss
        self.G_YtoXtoY_Loss.backward()

        self.G_Loss = self.G_XtoY_Loss + self.G_YtoX_Loss + self.G_XtoY_Cycle_Loss + self.G_YtoX_Cycle_Loss
        # self.G_Loss.backward()

    def train_d(self):
        self.d_optimizer.zero_grad()
        self.backward_d()
        self.d_optimizer.step()

    def train_g(self):
        self.g_optimizer.zero_grad()
        self.backward_g()
        self.g_optimizer.step()

    def train(self, data):
        self.load_data(data)
        self.forward()
        self.train_g()
        self.forward()
        self.train_d()

    def print_train_log(self, epoch, iteration):
        print('=' * 80)
        print('epoch:', epoch, '|| iteration:', iteration)
        print('D_X_Loss: {:f}'.format(self.D_X_Loss))
        print('D_Y_Loss: {:f}'.format(self.D_Y_Loss))
        print('D_Loss: {:f}'.format(self.D_Loss))
        print('G_XtoY_Loss: {:f}'.format(self.G_XtoY_Loss))
        print('G_XtoY_Cycle_Loss: {:f}'.format(self.G_XtoY_Cycle_Loss))
        # print('G_XtoY_identity_Loss: {:f}'.format(self.G_XtoY_identity_Loss))
        print('G_YtoX_Loss: {:f}'.format(self.G_YtoX_Loss))
        print('G_YtoX_Cycle_Loss: {:f}'.format(self.G_YtoX_Cycle_Loss))
        # print('G_YtoX_identity_Loss: {:f}'.format(self.G_YtoX_identity_Loss))
        print('G_Loss: {:f}'.format(self.G_Loss))
        print('=' * 80)

    def save_train_log(self, filename):
        if not os.path.exists(filename):
            header = [['D_X_Loss', 'D_Y_Loss', 'D_Loss', 'G_XtoY_Loss', 'G_XtoY_Cycle_Loss',
                      'G_YtoX_Loss', 'G_YtoX_Cycle_Loss', 'G_Loss']]
            with open(filename, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(header)
            csv_file.close()
        data = [float(self.D_X_Loss), float(self.D_Y_Loss), float(self.D_Loss), float(self.G_XtoY_Loss),
                float(self.G_XtoY_Cycle_Loss), float(self.G_YtoX_Loss), float(self.G_YtoX_Cycle_Loss),
                float(self.G_Loss)]
        with open(filename, mode='a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)
        csv_file.close()

    def test(self, data, index):
        self.load_data(data)
        self.forward()
        image_real_X = CycleGAN.tensor_to_numpy(self.image_real_X)
        # image_real_Y = CycleGAN.tensor_to_numpy(self.image_real_Y)
        # image_fake_X = CycleGAN.tensor_to_numpy(self.image_fake_X)
        image_fake_Y = CycleGAN.tensor_to_numpy(self.image_fake_Y)
        # image_reconstructed_X = CycleGAN.tensor_to_numpy(self.image_reconstructed_X)
        # image_reconstructed_Y = CycleGAN.tensor_to_numpy(self.image_reconstructed_Y)
        # images = [image_real_X, image_fake_Y, image_reconstructed_X, image_real_Y, image_fake_X, image_reconstructed_Y]
        CycleGAN.save_image(image_real_X, f'./images/real/test-X-{index}.png')
        # CycleGAN.save_image(image_real_Y, f'./samples/test-Y-{index}.png')
        # CycleGAN.save_image(image_fake_X, f'./samples/test-YtoX-{index}-{epoch}-{iteration}.png')
        CycleGAN.save_image(image_fake_Y, f'./images/fake/test-XtoY-{index}.png')
        # CycleGAN.save_image(image_reconstructed_X, f'./samples/test-XtoYtoX-{index}-{epoch}-{iteration}.png')
        # CycleGAN.save_image(image_reconstructed_Y, f'./samples/test-YtoXtoY-{index}-{epoch}-{iteration}.png')
        # CycleGAN.save_image(image_real_X, './samples/test-X.png')
        # CycleGAN.save_image(image_real_Y, './samples/test-Y.png')
        # CycleGAN.save_image(image_fake_X, './samples/test-YtoX.png')
        # CycleGAN.save_image(image_fake_Y, './samples/test-XtoY.png')
        # CycleGAN.save_image(image_reconstructed_X, './samples/test-XtoYtoX.png')
        # CycleGAN.save_image(image_reconstructed_Y, './samples/test-YtoXtoY.png')

        # fig = plt.figure(figsize=(10, 10))
        # rows, columns = 2, 3
        # ax = []
        # for i in range(1, columns * rows + 1):
        #     img = images[i-1]
        #     ax.append(fig.add_subplot(rows, columns, i))
        #     plt.axis('off')
        #     plt.imshow(img)
        # ax[0].set_title('input x')
        # ax[1].set_title('output G(x)')
        # ax[2].set_title('reconstruction F(G(x))')
        # ax[3].set_title('input y')
        # ax[4].set_title('output F(y)')
        # ax[5].set_title('reconstruction G(F(y))')
        # plt.savefig(f'./samples/test{index}-{epoch}.png')
        # plt.close()
