1. import necessary packages
2. use np.load to load train, dev dataset and convert it to tensor object
3. load MyDataset class for train & dev dataset
4. create dataloader with batch size 500 context/padding size 30
5. load model:

Model(
  (model): Sequential(
    (0): Linear(in_features=2440, out_features=3000, bias=True)
    (1): BatchNorm1d(3000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.1)
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=3000, out_features=2000, bias=True)
    (5): BatchNorm1d(2000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): LeakyReLU(negative_slope=0.1)
    (7): Dropout(p=0.2, inplace=False)
    (8): Linear(in_features=2000, out_features=1024, bias=True)
    (9): LeakyReLU(negative_slope=0.1)
    (10): Linear(in_features=1024, out_features=512, bias=True)
    (11): LeakyReLU(negative_slope=0.1)
    (12): Linear(in_features=512, out_features=71, bias=True)
  )
)
optimizer: torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
criterion: nn.CrossEntropyLoss()
6. load train & eval function for training and evaluation (15 epochs) and calculate accurate & avg loss
7. save model with largest evaluation accuracy after 14 epochs (epoch 13, starts from epoch 0)
8. load test dataset to make prediction and write csv file for kaggle
