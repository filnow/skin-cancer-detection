import torch
import torchmetrics
import lightning as L

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class SkinCancerDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, img_size: int = 224, 
                 train_dir: str = "./train", test_dir: str = "./test"):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_dir = train_dir
        self.test_dir = test_dir

        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size), antialias=True),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def prepare_data(self):
        self.trainData = datasets.ImageFolder(root=self.train_dir, transform=self.transform)
        self.testData = datasets.ImageFolder(root=self.test_dir, transform=self.transform)

    def setup(self, stage: str):
        if stage == 'fit':
            self.trainSet, self.valSet = torch.utils.data.random_split(self.trainData,\
                 [int(0.8 * len(self.trainData)), len(self.trainData) - int(0.8 * len(self.trainData))])
        
        if stage == 'test':
            self.testSet = self.testData   

    def train_dataloader(self):
        return DataLoader(self.trainSet, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.valSet, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.testSet, batch_size=self.batch_size, shuffle=False, num_workers=4)
    

class SkinCancerModule(L.LightningModule):
    def __init__(self, learning_rate: float = 0.001, num_classes: int = 2):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        output = self(inputs)
        loss = nn.CrossEntropyLoss()(output, labels)

        self.train_acc(output, labels)
        self.log('train_acc', self.train_acc, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        output = self(inputs)
        loss = nn.CrossEntropyLoss()(output, labels)

        self.val_acc(output, labels)
        self.log('val_acc', self.val_acc, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)

        return loss
    
    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        output = self(inputs)

        self.test_acc(output, labels)
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)
    

if __name__ == '__main__':
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 3
    MIN_DELTA = 0.01

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    dm = SkinCancerDataModule(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    dm.prepare_data()
    dm.setup('fit')

    model = SkinCancerModule(learning_rate=LEARNING_RATE, num_classes=len(dm.trainData.classes))
    
    early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=PATIENCE, verbose=True,\
         min_delta=MIN_DELTA, stopping_threshold=0.01, check_finite=True, check_on_train_epoch_end=False, mode="max")
    
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator=device, callbacks=[early_stop_callback])

    trainer.fit(model, dm)

    dm.setup('test')
    trainer.test(model, dm.test_dataloader())

    



