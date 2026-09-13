import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
# import torch.autograd as autograd

# autograd.set_detect_anomaly(True)
from datasets import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler

# cuda setup
import os
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True} 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']='1'
# hyper params
batch_size = 64
latent_size = 500
epochs = 10000


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=False, **kwargs)


dataset = LMDBDataset('vqvae-code/')
loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

train_loader = tqdm(loader)

def one_hot(labels, class_size , tensor , epoch):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        # targets[i, label] = torch.max(tensor)
         targets[i, label] = 0
         if(epoch>20):
              targets[i, label] = 0

    return targets.to(device)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        # self.fc0 = nn.Linear(feature_size, feature_size)
        self.fc1 = nn.Linear(feature_size, 1200)
        self.fc2 = nn.Linear(1200, 800)
        self.norm1 = nn.LayerNorm(800)
        self.fc21 = nn.Linear(800, latent_size)
        self.fc22 = nn.Linear(800, latent_size)
        
        # decode
        self.fc3 = nn.Linear(latent_size, 800)
        self.fc32 = nn.Linear(800, 1200)
        self.norm2 = nn.LayerNorm(1200)
        self.fc4 = nn.Linear(1200, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def encode(self, x, c):
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        # Scale up class vector to feature size and add element-wise
        c_expanded = c.repeat(1, 245)
        tensor2 = torch.tensor([[0,0,0,0,0]]).to(device)
        c_expanded = torch.cat((c_expanded, tensor2), dim=1)
        combined = x + c_expanded
        # h0 = self.elu(self.fc0(combined))
        h1 = self.elu(self.fc1(combined))
        h2 = self.norm1(self.elu(self.fc2(h1)))
        z_mu = self.fc21(h2)
        z_var = self.fc22(h2)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        # inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        inputs = z
        h3 = self.elu(self.fc3(inputs))
        
         
        # h3 = torch.cat([h3, c], 1)
        h4 = self.norm2(self.elu(self.fc32(h3)))
        # h4 = torch.cat([h4, c], 1)
        return self.relu(self.fc4(h4))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 1720), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# create a CVAE model
model = CVAE(1720, latent_size, 7).to(device)
model_path = r'checkpoint\cvae_improved\checkpoint_cvae_improved_no fl 500_6000.pt'
model.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(model.parameters(), lr=5e-7)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1720), reduction='sum')
    recon_x = min_max_normalize(recon_x)*511
    x = min_max_normalize(x)*511
    # recon_x = torch.clamp(recon_x, 0, 511)
    # recon_x = recon_x / math.sqrt(1720)
    # x = x / math.sqrt(1720)
    # print(recon_x)
    MSE = F.mse_loss(recon_x, x.view(-1, 1720), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def min_max_normalize(tensor):
    """
    Performs min-max normalization on a tensor in-place.

    Args:
        tensor (torch.Tensor): The tensor to be normalized.

    Modifies the input tensor in-place.
    """

    min_val = torch.min(tensor)
    # print (min_val)
    max_val = torch.max(tensor)
    # print (max_val)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor



def train(epoch):
    model.train()
    train_loss = 0
    # print("enter")
    
    for i, (data, labels, salience, file_name) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        flattened_tensor = data.view(-1).float()
        labels = one_hot(labels, 7, flattened_tensor, epoch)

        # Scale up class vector to feature size and add element-wise
        labels_expanded = labels.repeat(1, 245)
        tensor2 = torch.tensor([[0,0,0,0,0]]).to(device)
        concatenated_tensor = torch.cat((labels_expanded, tensor2), dim=1)
        # for i in range(concatenated_tensor[:,]):
        combined_input = flattened_tensor - concatenated_tensor

        # print("Till here")
        # combined_input = min_max_normalize(combined_input)
        # flattened_tensor = min_max_normalize(flattened_tensor)
        # if(epoch<10):
        #     combined_input = combined_input/5
        #     flattened_tensor = flattened_tensor/5
        recon_batch, mu, logvar = model(combined_input, labels)
        optimizer.zero_grad()
        # recon_batch.clamp(0, 1)
        # recon_batch = torch.clamp(recon_batch, 0, 1)
        # recon_batch = min_max_normalize(recon_batch)
        # print(recon_batch)
        loss = loss_function(recon_batch, flattened_tensor, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        # if batch_idx % 20 == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, 1 * len(flattened_tensor), len(train_loader),
        #         100. * 1 / len(train_loader),
        #         loss.item() / len(flattened_tensor)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))
    
    if epoch%100 == 0:
        checkpoint_path = f"./checkpoint/cvae_improved/checkpoint_cvae_improved_no fl 500_{epoch}.pt"

    # Save the model state dictionary (checkpoint)
        torch.save(model.state_dict(), checkpoint_path)


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, labels) in enumerate(test_loader):
#             data, labels = data.to(device), labels.to(device)
#             labels = one_hot(labels, 10)
#             recon_batch, mu, logvar = model(data, labels)
#             test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
#             if i == 0:
#                 n = min(data.size(0), 5)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(-1, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'reconstruction_' + str(f"{epoch:02}") + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

def train_epoch():
    for epoch in range(6001, epochs + 1):
                train(epoch)
            # test(epoch)
            # with torch.no_grad():
            #     c = torch.eye(10, 10).cuda()
            #     sample = torch.randn(10, latent_size).to(device)
            #     sample = model.decode(sample, c).cpu()
                # save_image(sample.view(10, 1, 28, 28),
                #            'sample_' + str(f"{epoch:02}") + '.png')

train_epoch()