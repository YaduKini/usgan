import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from loader import get_ultrasound_data, get_generator_input
from ultrasoundGAN import weights_init, Generator, Discriminator, Generator2, Discriminator2

params = {
    "bsize": 5,  # Batch size during training.
    'imsize': 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc': 3,  # Number of channles in the training images. For coloured images this is 3.
    'nz': 12288,  # Size of the Z latent vector (the input to the generator). 12288
    'ngf': 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    'ndf': 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs': 2,  # Number of training epochs.
    'lr': 0.0002,  # Learning rate for optimizers
    'beta1': 0.5,  # Beta1 hyperparam for Adam optimizer
    'save_epoch': 2}  # Save step.

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# device = torch.device("cpu")

# Get the data.
dataloader = get_ultrasound_data(params)

# Plot the training images.
# sample_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))
#
# plt.show()

# Create the generator.
netG = Generator(params).to(device)
netG2 = Generator2(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
netG2.apply(weights_init)
# Print the model.
# print("Generator Stage 1: {}".format(netG))
# print("Generator Stage 2: {}".format(netG2))

# Create the discriminator.
netD = Discriminator(params).to(device)
netD2 = Discriminator2(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
netD2.apply(weights_init)
# Print the model.
# print("Discriminator Stage 1: {}".format(netD))
# print("Discriminator Stage 2: {}".format(netD2))
# Binary Cross Entropy loss function.
criterion = nn.BCELoss()

criterion2 = nn.BCELoss()

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

# fixed image batch input to Stage 1 Generator
sample_dataloader = get_generator_input(params)
fixed_noise = next(iter(sample_dataloader))
# print("length of fixed noise: {}".format(len(fixed_noise)))

fixed_noise = fixed_noise[0].to(device)
# print("shape of fixed noise 1 before reshape: {}".format(fixed_noise.size()))
fixed_noise = torch.reshape(fixed_noise, (params['bsize'], params['nz'], 1, 1))
# print("shape of fixed noise 1 after reshape: {}".format(fixed_noise.size()))

real_label = 1
fake_label = 0

# Optimizer for the discriminator stage 1.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the discriminator stage 2.
optimizerD2 = optim.Adam(netD2.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Optimizer for the generator stage 1
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the generator stage 2
optimizerG2 = optim.Adam(netG2.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
img_list2 = []
final_imglist = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

# Stores generator losses during training.
# G2_losses = []
# Stores discriminator losses during training.
# D2_losses = []
fake_data_list = []
final_image_list = []
iters = 0

print("Starting Training Loop...")

for epoch in range(params['nepochs']):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_data.size(0)

        # Make accumalated gradients of the discriminator zero.
        netD.zero_grad()
        netD2.zero_grad()
        # Create labels for the real data. (label=1)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_data).view(-1)
        output2 = netD2(real_data).view(-1)
        errD_real = criterion(output, label)
        errD2_real = criterion2(output2, label)
        errD_real_total = errD_real + errD2_real
        # Calculate gradients for backpropagation.
        errD_real_total.backward()
        D_x = output.mean().item()
        D2_x = output2.mean().item()

        # Sample random data from a unit normal distribution.
        # noise = torch.randn(b_size, params['nz'], 1, 1, device=device)

        # stage 1 generator input
        noise = next(iter(sample_dataloader))
        noise = noise[0].to(device)
        noise = torch.reshape(noise, (b_size, params['nz'], 1, 1))
        # print("size of generator input data: {}".format(noise.size()))

        # Generate fake data (images).
        fake_data = netG(noise)

        # create tmp variable for plotting
        fake_data_tmp = netG(noise).detach().cpu()
        fake_data_list.append(vutils.make_grid(fake_data_tmp, padding=2, normalize=True))

        fake_data_copy = torch.reshape(fake_data, (b_size, params['nz'], 1, 1))
        fake_data_stage2 = netG2(fake_data_copy)

        # create tmp variable for plotting
        fake_data_stage2_tmp = netG2(fake_data_copy).detach().cpu()

        # fake data generated from stage 2
        final_image_list.append(vutils.make_grid(fake_data_stage2_tmp, padding=2, normalize=True))

        # fake data generated from stage 1

        # fake_data_copy_tmp = netG2(fake_data2).detach().cpu()
        # final_imglist.append(vutils.make_grid(fake_data_copy_tmp, padding=2, normalize=True))

        # fake_data_copy2 = torch.reshape(fake_data_copy, (b_size, params['nz'], 1, 1))
        # print("size of fake data copy: {}".format(fake_data_copy.size()))
        # print("type of fake data: {}".format(type(fake_data)))

        # fake_data2 = netG2(fake_data_copy)
        # fake_data_copy = torch.reshape(fake_data_copy, (b_size, params['nz'], 1, 1))
        # fake_data = torch.reshape(fake_data, (b_size, params['nz'], 1, 1))

        # print("type of fake data 2: {}".format(type(fake_data_copy)))

        # this fake data 2 will be input to stage 2 generator

        # img_list2.append(fake_data)
        # noise2 = torch.reshape(noise, (1, params['nz']*5, 1, 1))
        # fake_data2 = netG(noise2)
        # img_list2.append(fake_data2)
        # print("type of fake data is: {}".format(type(fake_data)))
        # print("length of fake data is: {}".format(len(fake_data)))
        # Create labels for fake data. (label=0)
        label.fill_(fake_label)
        # Calculate the output of the discriminator of the fake data.
        # As no gradients w.r.t. the generator parameters are to be
        # calculated,
        # detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        output = netD(fake_data.detach()).view(-1)
        output2 = netD2(fake_data_stage2.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD2_fake = criterion2(output2, label)
        # Calculate gradients for backpropagation.
        errD_fake_total = errD_fake + errD2_fake
        errD_fake_total.backward()
        D_G_z1 = output.mean().item()
        D2_G_z1 = output2.mean().item()

        # Net discriminator loss.
        errD_total = errD_real_total + errD_fake_total
        # Update discriminator parameters.
        optimizerD.step()
        optimizerD2.step()

        # Make accumalted gradients of the generator zero.
        netG.zero_grad()
        netG2.zero_grad()
        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        label.fill_(real_label)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        output = netD(fake_data).view(-1)
        output2 = netD2(fake_data_stage2).view(-1)

        factor = 0.0001
        l1_loss = nn.L1Loss(reduction='sum')
        l1_reg_loss = factor * l1_loss(output, label)
        # print("self regularization loss: {}".format(l1_reg_loss))
        errG = criterion(output, label) + l1_reg_loss
        # for stage 2
        l1_loss_stage2 = nn.L1Loss(reduction='sum')
        l1_reg_loss_stage2 = factor * l1_loss_stage2(output2, label)
        errG2 = criterion2(output2, label) + l1_reg_loss_stage2
        errG_total = errG + errG2
        # errG = criterion(output, label)
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()
        errG_total.backward()

        D_G_z2 = output.mean().item()
        D2_G_z2 = output2.mean().item()
        # Update generator parameters.
        optimizerG.step()
        optimizerG2.step()

        # Check progress of training.
        if i % 50 == 0:
            print(torch.cuda.is_available())
            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD2(x): %.4f\tD(G(z)): %.4f / %.4f\tD2(G2(z)): %.4f / %.4f'
                % (epoch, params['nepochs'], i, len(dataloader),
                   errD_total.item(), errG_total.item(), D_x, D2_x, D_G_z1, D_G_z2, D2_G_z1, D2_G_z2))

        # Save the losses for plotting.
        G_losses.append(errG_total.item())
        D_losses.append(errD_total.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 50 == 0) or ((epoch == params['nepochs'] - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
                # for stage 2 use fake_data2 = netG2(fake_data)
                # final_image = netG2(fake_data2).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))
            img_list2.append(fake_data)
            # final_imglist.append(final_image)

        iters += 1
        # print("length of img list 2: {}".format(len(img_list2)))
        # print("length of final img list : {}".format(len(final_imglist)))

#     # Save the model.
#     if epoch % params['save_epoch'] == 0:
#         torch.save({
#             'generator' : netG.state_dict(),
#             'generator2' : netG2.state_dict(),
#             'discriminator' : netD.state_dict(),
#             'discriminator2': netD2.state_dict(),
#             'optimizerG' : optimizerG.state_dict(),
#             'optimizerG2': optimizerG2.state_dict(),
#             'optimizerD' : optimizerD.state_dict(),
#             'optimizerD2': optimizerD2.state_dict(),
#             'params' : params
#             }, 'model/model_epoch_{}.pth'.format(epoch))

# # Save the final trained model.
# torch.save({
#             'generator' : netG.state_dict(),
#             'generator2' : netG2.state_dict(),
#             'discriminator' : netD.state_dict(),
#             'discriminator2' : netD2.state_dict(),
#             'optimizerG' : optimizerG.state_dict(),
#             'optimizerG2' : optimizerG2.state_dict(),
#             'optimizerD' : optimizerD.state_dict(),
#             'optimizerD2' : optimizerD2.state_dict(),
#             'params' : params
#             }, 'model/model_final.pth')


# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# for i in range(len(img_list2)):
#     plt.figure(figsize=(16, 16))
#     plt.axis("off")
#     plt.title("Generated Images for iteration: {}".format(i))
#     plt.imshow(np.transpose(vutils.make_grid(img_list[i].to(device)[:64], padding=2, normalize=True).cpu(),(1, 2, 0)))

print("length of image list 2: {}".format(len(img_list2)))
print("length of final image list : {}".format(len(final_imglist)))
