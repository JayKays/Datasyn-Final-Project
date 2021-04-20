
import os
import torch

from matplotlib import pyplot as plt
from DatasetLoader import make_test_dataloader, make_train_dataloaders
from plotting import plot_segmentation
from config import *
from utils import to_cuda


def test(model, dataset, bs = 20):

    if dataset == 'CAMUS_resized':
        _ , _ , test_data = make_train_dataloaders(dataset, (300,100,50))
    else:
        test_data = make_test_dataloader(dataset)
    
    xb, yb = next(iter(test_data))
    
    xb, yb = xb[:bs], yb[:bs]

    with torch.no_grad():
        predb = model(xb.cuda())
    
    plot_segmentation(xb, yb, predb)
    plt.show()


if __name__ == "__main__":

    model_name = 'Default_UNET'

    model_path = os.path.join(SAVE_DIR,'Saved_models', model_name)

    newest_file = newest_checkpoint()
    newest_model = torch.load(newest_file)

    unet.load_state_dict(newest_model["model"])
    # opt.load_state_dict(newest_model["optimizer"])

    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    start_epoch = newest_model["epoch"] + 1
    start_loss = newest_model["loss"]