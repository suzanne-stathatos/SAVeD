import argparse
import logging
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
# from torch.utils.tensorboard import SummaryWriter

import data, models, utils


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    utils.setup_experiment(args)
    utils.init_logging(args)
    wandb.init(
        project="udvd-denoising",
        config=vars(args),
        name=args.experiment_dir.split('/')[-1]  # Use experiment name as run name
    )

    # Build data loaders, a model and an optimizer
    model = models.build_model(args).to(device)
    cpf = model.c # channels per frame
    mid = args.n_frames // 2
    model = nn.DataParallel(model)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25, 30], gamma=0.5)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
    else:
        global_step = -1
        start_epoch = 0

    train_loader, valid_loader = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size, image_size=args.image_size, stride=args.stride, n_frames=args.n_frames) # , stride=args.stride

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr", "valid_ssim"])}
    # writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    for epoch in range(start_epoch, args.num_epochs):
        if args.resume_training:
            if epoch %10 == 0:
                optimizer.param_groups[0]["lr"] /= 2
                print('learning rate reduced by factor of 2')

        # train_bar = utils.ProgressBar(train_loader, epoch)
        # replace ProgressBar with tqdm
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch:02d}', total=len(train_loader), leave=True)
        for meter in train_meters.values():
            meter.reset()
        
        model.train()

        try: 
            for batch_id, inputs in enumerate(train_bar):
                try:
                    global_step += 1
                    noisy_inputs = inputs = inputs.to(device)

                    full_outputs = model(noisy_inputs)
                    # get first element of outputs
                    outputs = out = full_outputs[0]

                    noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

                    # take the first channel of outputs to make it match with the inputs
                    out = out[:, 0, :, :]
                    outputs = outputs[:, 0, :, :]
                    # add back channel dimension
                    out = out.unsqueeze(1)
                    outputs = outputs.unsqueeze(1)

                    loss = utils.loss_function(out, noisy_frame, mode=args.loss, sigma=args.noise_std, device=device)

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_psnr = utils.psnr(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs)
                    train_ssim = utils.ssim(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs)
                    train_meters["train_loss"].update(loss.item())
                    train_meters["train_psnr"].update(train_psnr.item())
                    train_meters["train_ssim"].update(train_ssim.item())

                    # train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)
                    tqdm.write(f"Epoch {epoch:02d} | Batch {batch_id:05d}/{len(train_loader):05d} | "
                             f"Loss: {loss.item():.3f} | PSNR: {train_psnr.item():.3f} | "
                             f"SSIM: {train_ssim.item():.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

                    if global_step % args.log_interval == 0:
                        # Replace tensorboard logging with wandb
                        wandb.log({
                            "lr": optimizer.param_groups[0]["lr"],
                            "loss/train": loss.item(),
                            "psnr/train": train_psnr.item(),
                            "ssim/train": train_ssim.item(),
                            "global_step": global_step
                        })
                        
                        # Log gradients
                        gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                        wandb.log({
                            "gradients": wandb.Histogram(gradients.cpu().numpy()),
                            "global_step": global_step
                        })
                        sys.stdout.flush()

                    if (batch_id+1) % 250 == 0:
                        # logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")
                        tqdm.write(f" | {batch_id+1} mini-batches ended")
                    if (batch_id+1) % 100000 == 0:
                        model.eval()
                        for meter in valid_meters.values():
                            meter.reset()

                        valid_bar = utils.ProgressBar(valid_loader)
                        running_valid_psnr = 0.0
                        for sample_id, sample in enumerate(valid_bar):
                            if sample_id > 50:
                                break
                            with torch.no_grad():
                                noisy_inputs = sample = sample.to(device)

                                outputs = out = model(noisy_inputs)[0]
                                noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

                                # take the first channel of outputs to make it match with the inputs
                                out = out[:, 0, :, :]
                                outputs = outputs[:, 0, :, :]
                                # add back channel dimension
                                out = out.unsqueeze(1)
                                outputs = outputs.unsqueeze(1)

                                valid_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs)
                                valid_ssim = utils.ssim(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs)
                                running_valid_psnr += valid_psnr
                                valid_meters["valid_psnr"].update(valid_psnr.item())
                                valid_meters["valid_ssim"].update(valid_ssim.item())
                        running_valid_psnr /= (sample_id+1)

                        # Replace tensorboard logging with wandb
                        wandb.log({
                            "psnr/valid": valid_meters['valid_psnr'].avg,
                            "ssim/valid": valid_meters['valid_ssim'].avg,
                            "global_step": global_step
                        })
                        sys.stdout.flush()

                        # logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
                        logging.info(tqdm.write(f"EVAL: {valid_meters['valid_psnr'].avg:.3f}"))
                        utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_psnr"].avg, mode="max")
                        print("checkpoint saved to", args.experiment_dir)
                except Exception as e:
                    print(f"Error in batch {batch_id}: {str(e)}")
                    continue  # Skip problematic batch and continue training
        except Exception as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            # Optionally save checkpoint before continuing
            utils.save_checkpoint(args, global_step, model, optimizer, 
                               score=valid_meters["valid_psnr"].avg, mode="max")
            continue  # Move to next epoch

        scheduler.step()

        logging.info(tqdm.write(f"Epoch {epoch:02d} | Train: {train_meters['train_psnr'].avg:.3f} | Valid: {valid_meters['valid_psnr'].avg:.3f}"))
        # logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"])))

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")
    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", help="path to data directory", required=True)
    parser.add_argument("--dataset", default="CFC", help="train dataset name")
    parser.add_argument("--batch-size", default=4, type=int, help="train batch size")
    parser.add_argument("--image-size", default=256, type=int, help="image size for train")
    parser.add_argument("--n-frames", default=5, type=int, help="number of frames for training")
    parser.add_argument("--stride", default=64, type=int, help="stride for patch extraction")

    # Add model arguments
    parser.add_argument("--model", default="blind-video-net-4", help="model architecture")

    # Add loss function
    parser.add_argument("--loss", default="mse", help="loss function used for training")

    # Add noise arguments
    parser.add_argument("--noise_dist", default="G", help="G - Gaussian, P - Poisson")
    parser.add_argument("--noise_mode", default="S", help="B - Blind, S - one noise level")
    parser.add_argument('--noise_std', default = 30, type = float,
                        help = 'noise level when mode is S')
    parser.add_argument('--min_noise', default = 5, type = float,
                        help = 'minimum noise level when mode is B')
    parser.add_argument('--max_noise', default = 50, type = float,
                        help = 'maximum noise level when mode is B')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=1, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=50, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Comment this out if running for the first time
    args.resume_training = True
    main(args)
