import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from models import MultitaskGPModel

x = torch.linspace(-4,4,100).reshape(-1,1)
y_1 = torch.cos(x) + torch.randn_like(x) * 0.2
y_2 = torch.sin(x) + torch.randn_like(x) * 0.2
y = torch.cat([y_1, y_2], dim=1)

train_n = len(x)
train_x = x[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = x[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()




number_inducing_points = 5
inducing_points = torch.linspace(-2,2,number_inducing_points).reshape(-1,1)
model = MultitaskGPModel(inducing_points=inducing_points, num_tasks=y.size(1), batch_indipendent_kernel=True)




num_epochs = 1000

model.train()

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.size(1))
# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)




if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
    train_x, train_y = train_x.cuda(), train_y.cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()

model.convert_to_exact_gp()

posterior_f, std_f=model.posterior_f(train_x, return_std=True)
posterior_f_prime, std_f_prime=model.posterior_f_prime(train_x, return_std=True)
posterior_f_prime=posterior_f_prime.squeeze()
std_f_prime=std_f_prime.squeeze()
# print("posterior_f")
# print(posterior_f.shape)
# print("std_f")
# print(std_f.shape)
# print("posterior_f_prime")
# print(posterior_f_prime.shape)
# print("std_f_prime")
# print(std_f_prime.shape)

# plot the data and the derivatives
with torch.no_grad():
    plt.scatter(train_x.cpu().numpy(), train_y[:,0].cpu().numpy(), label='Train')
    plt.plot(train_x.cpu().numpy(), posterior_f[0,:].cpu().numpy(), label='Posterior f')
    plt.fill_between(train_x.cpu().numpy().reshape(-1), (posterior_f[0,:]-std_f[0,:]).cpu().numpy().reshape(-1), (posterior_f[0,:]+std_f[0,:]).cpu().numpy().reshape(-1), alpha=0.5)
    plt.scatter(model.x_inducing.cpu().numpy(), model.y_inducing[0].cpu().numpy(), label='Inducing points')

    plt.plot(train_x.cpu().numpy(), posterior_f_prime[0,:].cpu().numpy(), label='Posterior f prime')
    plt.fill_between(train_x.cpu().numpy().reshape(-1), (posterior_f_prime[0,:]-std_f_prime[0,:]).cpu().numpy().reshape(-1), (posterior_f_prime[0,:]+std_f_prime[0,:]).cpu().numpy().reshape(-1), alpha=0.5)
    plt.legend()
    plt.figure()
    plt.scatter(train_x.cpu().numpy(), train_y[:,1].cpu().numpy(), label='Train')
    plt.plot(train_x.cpu().numpy(), posterior_f[1,:].cpu().numpy(), label='Posterior f')
    plt.fill_between(train_x.cpu().numpy().reshape(-1), (posterior_f[1,:]-std_f[1,:]).cpu().numpy().reshape(-1), (posterior_f[1,:]+std_f[1,:]).cpu().numpy().reshape(-1), alpha=0.5)
    plt.plot(train_x.cpu().numpy(), posterior_f_prime[1,:].cpu().numpy(), label='Posterior f prime')
    plt.fill_between(train_x.cpu().numpy().reshape(-1), (posterior_f_prime[1,:]-std_f_prime[1,:]).cpu().numpy().reshape(-1), (posterior_f_prime[1,:]+std_f_prime[1,:]).cpu().numpy().reshape(-1), alpha=0.5)
    plt.scatter(model.x_inducing.cpu().numpy(), model.y_inducing[1].cpu().numpy(), label='Inducing points')
    plt.show()

