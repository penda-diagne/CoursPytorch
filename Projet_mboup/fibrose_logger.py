class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
       
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
