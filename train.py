from AnimeGAN import Trainer
import argparse

DATA_SET = "realistic_cartoon"
BATCH_SIZE = 2
EPOCH = 50
START_EPOCH = 1
INIT_G_EPOCH = 5
INIT_G_LR = 2e-4
G_LR = 1e-4
D_LR = 1e-4
DEVICE = "cuda"
IMAGE_SIZE = 512
OUT_DIR = "./realistic_cartoon_outputs"

def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANv3")

    parser.add_argument('--dataset', type=str, default=DATA_SET, help='dataset_name')
    parser.add_argument('--init_G_epoch', type=int, default=INIT_G_EPOCH, help='The number of epochs for generator initialization')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default=START_EPOCH, help='The beginning index of training epoch')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The size of batch size')
    parser.add_argument('--init_G_lr', type=float, default=INIT_G_LR, help='The generator learning rate')
    parser.add_argument('--g_lr', type=float, default=G_LR, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=D_LR, help='The learning rate')
    parser.add_argument('--device', type=str, default=DEVICE, help="CPU or CUDA")
    parser.add_argument('--img_size', type=int, default=IMAGE_SIZE, help="image size")
    parser.add_argument('--out_dir', type=str, default=OUT_DIR, help="output dir")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(
        dataset=args.dataset,
        epoch=args.epoch,
        start_epoch=args.start_epoch,
        batch=args.batch_size,
        init_g_epoch=args.init_G_epoch,
        init_lr_g=args.init_G_lr,
        lr_g=args.g_lr,
        lr_d=args.d_lr,
        device=args.device,
        img_size=args.img_size,
        out_dir=args.out_dir,
    )
    trainer.train()