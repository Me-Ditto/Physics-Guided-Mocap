import torch
import os
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from modules import init, DatasetLoader, ModelLoader, LossLoader, seed_worker, set_seed
from utils.logger import savefig

###########global parameters#########
import sys
sys.argv = ['','--config=cfg_files\\demo.yaml']

def main(**args):
    seed = 7
    set_seed(seed)
    # global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'),type='cuda')
    viz = args.get('viz')
    mode = args.get('mode')
    g = torch.Generator()
    g.manual_seed(seed)

    # init project setting
    out_dir, logger, smpl = init(dtype=dtype, device=device, **args)

    # load loss function
    loss = LossLoader(device=device, **args)

    # load model
    model = ModelLoader(device=device, output=out_dir, smpl=smpl, **args)

    # create data loader
    dataset = DatasetLoader(smpl_model=smpl, dtype=dtype, **args)
    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )


    task = args.get('task')
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):

        # testing mode
        if epoch == 0 and mode == 'test':
            training_loss = -1.
            testing_loss = eval('%s_test' %task)(model, loss, test_loader, epoch, viz=viz, device=device)

        lr = model.optimizer.state_dict()['param_groups'][0]['lr']
        logger.append([int(epoch + 1), lr, training_loss, testing_loss])

    logger.close()


if __name__ == "__main__":
    args = parse_config()
    main(**args)





