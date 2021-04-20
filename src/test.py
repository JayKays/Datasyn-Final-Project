
import os

from config import *



def test(model, dataset_dir):
    pass

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