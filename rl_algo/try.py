import torch
 
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 1
 
# Create random Tensors to hold inputs and outputs.
x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
y = torch.sum(x, dim=1)
 
# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(size_average=False)
 
# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
  # Forward pass: compute predicted y by passing x to the model.
  y_pred = model(x).view(-1)
 
  # Compute and print loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.item())
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
