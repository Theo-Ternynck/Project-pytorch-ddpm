import os
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import trange

from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from helpers import infiniteloop, warmup_lr
from model import UNet

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("ch", 128, help="base channel of UNet")
flags.DEFINE_multi_integer("ch_mult", [1, 2, 2, 2], help="channel multiplier")
flags.DEFINE_multi_integer("attn", [1], help="add attention to these levels")
flags.DEFINE_integer("num_res_blocks", 2, help="# resblock in each level")
flags.DEFINE_float("dropout", 0.1, help="dropout rate of resblock")
# Gaussian Diffusion
flags.DEFINE_float("beta_1", 1e-4, help="start beta value")
flags.DEFINE_float("beta_T", 0.02, help="end beta value")
flags.DEFINE_integer("T", 1000, help="total diffusion steps")
flags.DEFINE_enum(
    "mean_type", "epsilon", ["xprev", "xstart", "epsilon"], help="predict variable"
)
flags.DEFINE_enum(
    "var_type", "fixedlarge", ["fixedlarge", "fixedsmall"], help="variance type"
)
# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 800000, help="total training steps")
flags.DEFINE_integer("img_size", 32, help="image size")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
# Logging & Sampling
flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", help="log directory")
flags.DEFINE_integer("sample_size", 64, "sampling size of images")
flags.DEFINE_integer("sample_step", 1000, help="frequency of sampling")
# Evaluation
flags.DEFINE_integer(
    "save_step",
    5000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer(
    "num_images", 50000, help="the number of generated images for evaluation"
)
flags.DEFINE_bool("fid_use_torch", False, help="calculate IS and FID on gpu")
flags.DEFINE_string("fid_cache", "./stats/cifar10.train.npz", help="FID cache")

flags.DEFINE_string("data_dir", "./data/", help="data directory")
flags.DEFINE_bool(
    "from_scratch",
    False,
    help="restart fine tuning from a certain state (optim, scheduler,model)",
)

flags.DEFINE_bool(
    "generation",
    True,
    help="wether or not to generate a batch of 256 new images after the training",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune(model, data_dir):

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
    )

    datalooper = infiniteloop(dataloader)

    if not FLAGS.from_scratch:
        ckpt = torch.load(os.path.join(FLAGS.logdir, "ckpt_fine_tune.pt"))
        model.to(device)
        model.load_state_dict(ckpt["net_model"])

    else:
        model.to(device)
        prev_step = 0

    # model setup
    optim = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, FLAGS.warmup)
    )

    if not FLAGS.from_scratch:
        optim.load_state_dict(state_dict=ckpt["optim"])
        sched.load_state_dict(state_dict=ckpt["sched"])
        x_T = ckpt["x_T"].to(device)
        prev_step = ckpt["step"]

    trainer = GaussianDiffusionTrainer(model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(
        device
    )
    net_sampler = GaussianDiffusionSampler(
        model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, "sample_fine_tune"), exist_ok=True)
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][: FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image("real_sample", grid)
    writer.flush()

    # backup all arguments : all flags used are stored in the file
    with open(os.path.join(FLAGS.logdir, "flagfile_fine_tune.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())

    # display model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            # log
            writer.add_scalar("loss", loss, step)
            pbar.set_postfix(loss="%.3f" % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                model.eval()
                with torch.no_grad():
                    x_0 = net_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir + "/sample_fine_tune",
                        "%d.png" % (step + prev_step),
                    )
                    print(path)
                    save_image(grid, path)
                    writer.add_image("sample_fine_tune", grid, step)
                model.train()

            # save
            if (
                FLAGS.save_step > 0 and (step + prev_step) % FLAGS.save_step == 0
            ) or step == FLAGS.total_steps - 1:
                ckpt = {
                    "net_model": model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step + prev_step,
                    "x_T": x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, "ckpt_fine_tune.pt"))

        if FLAGS.generation:
            images = generate_sample(net_sampler, model)
            save_image(
                torch.tensor(images[: min(256, FLAGS.num_images)]),
                os.path.join(FLAGS.logdir, "sample_fine_tune.png"),
                nrow=16,
            )

    writer.close()


def generate_sample(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    return images


def main(argv):
    # suppress a  nnoying inception_v3 initialization warning

    model = UNet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
    )

    warnings.simplefilter(action="ignore", category=FutureWarning)
    fine_tune(model, FLAGS.data_dir)


if __name__ == "__main__":
    app.run(main)
