# See https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss
# !pip install pytorch-metric-learning -q

from pytorch_metric_learning.losses import NTXentLoss

loss_func = NTXentLoss(temperature=0.10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Use a large batch size (might lead to RAM issues)
# Free Colab Version has ~ 12 GB of RAM
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
