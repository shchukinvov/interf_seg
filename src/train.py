from dataset import init_dataloaders
from interf_seg.data.interf_generator import init_interf_generator
from utils import *
from interf_seg.configs.collection import module_collection
from interf_seg.configs.experiment import *


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main(model, criterion, optimizer, scheduler, seed, num_epochs, batch_size):
    seed_all(seed)
    model_ = module_collection[model["name"]](**model["params"]).to(DEVICE)
    if not TRAINING or PRETRAIN:
        model_.load_state_dict(torch.load(os.path.join(CP_PATH, LOAD_CP)))

    if TRAINING:
        criterion_ = module_collection[criterion["name"]](**criterion["params"])
        optimizer_ = module_collection[optimizer["name"]](model_.parameters(), **optimizer["params"])
        scheduler_ = module_collection[scheduler["name"]](optimizer_, **scheduler["params"])
        dataloaders_ = init_dataloaders(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TRAIN_TRANSFORMS, VAL_TRANSFORMS,
                                        test_size=0.05,
                                        batch_size=batch_size,
                                        random_state=seed,
                                        num_workers=2)
        train_loss, val_loss = train_fn(model_, dataloaders_, criterion_, optimizer_, num_epochs, DEVICE, scheduler_,
                                        validation=True)
        # dataloaders_ = init_interf_generator(size=10000, shape=400, batch_size=batch_size, num_workers=2)
        # train_loss = train_fn(model_, dataloaders_, criterion_, optimizer_, num_epochs, DEVICE, scheduler_, validation=False)
        if SAVE_MODEL:
            ps = ExperimentLogger(EXPERIMENT_LIST)
            model_path = "model_" + str(len(ps)) + ".pth"
            torch.save(model_.state_dict(), os.path.join(CP_PATH, model_path))
            val_scores = validate_model(model_, postprocessing=False, save_predictions=False, show_results=True)
            ps.parse_and_log(val_scores, model_path, **EXPERIMENT_PARAMS)
        show_losses(train_loss, val_loss)

    else:
        im = np.array(Image.open("D:/ML_Projects/interf_seg/data/link_im2.bmp").convert("L"))
        # show_model_preds(model_, im)
        show_model_featuremap(model_, im)
        # validate_model(model_, postprocessing=True, save_predictions=False, show_results=True)


if __name__ == "__main__":
    main(**EXPERIMENT_PARAMS)
