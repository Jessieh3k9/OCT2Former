from settings_args import *
import torch.nn.functional as F
from torch.backends import cudnn
from utils import utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import os
from utils.loss import DiceLoss
from tqdm import tqdm
import csv
import random
import numpy as np
from PIL import Image
import time
import torchsummary
from torchvision.utils import save_image, make_grid
from DNN_printer import DNN_printer
from model.OCT2Former import OCT2Former
from model.swinunet import SwinTransformerSys
from model.TransUNet.TransUNet import get_transNet
import sys

sys.setrecursionlimit(100000)

def build_model(args):
    if args.network == "OCT2Former":
        return OCT2Former(in_chans=args.in_channel, num_classes=args.n_class,
                embed_dims=args.vit_dims, k=args.token_dim,
                num_heads=[2, 4, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
                depths=args.depths, aux=args.aux, spec_inter=args.spec_interpolation)
    if args.network == "swinunet":
        crop_h, crop_w = args.crop_size
        swin_img_size = ((max(crop_h, crop_w) + 223) // 224) * 224
        return SwinTransformerSys(img_size=swin_img_size, patch_size=4, in_chans=args.in_channel, num_classes=args.n_class)
    if args.network == "TransUNet":
        return get_transNet(args.n_class)
    raise ValueError(f"Unsupported network: {args.network}")


def normalize_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs
    return {"main_out": outputs}


def format_duration(seconds):
    if seconds is None:
        return "--:--:--"

    total_seconds = max(int(seconds), 0)
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main(args, num_fold=0):
    torch.set_num_threads(1)
    model = build_model(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.mode == "train":
        train(model, device, args, num_fold=num_fold)

    elif args.mode == "test":
        return test(model, device, args, num_fold=num_fold)

    else:
        raise NotImplementedError



def train(model, device, args, num_fold=0):
    dataset_train = myDataset(args.data_root, args.target_root, args.crop_size,  "train",
                                 k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold,  data_root_aux=args.data_root_aux, img_aug=args.img_aug)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True) 
    num_train_data = len(dataset_train)
    dataset_val = myDataset(args.data_root, args.target_root, args.crop_size, "val",
                               k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold, data_root_aux=args.data_root_aux)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True) 
    num_train_val = len(dataset_val)  
    ####################################################################################################################
    writer = SummaryWriter(log_dir=args.log_dir[num_fold], comment=f'tb_log')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(torch.tensor(args.class_weight, device=device))
    criterion_dice = DiceLoss()

    cp_manager = utils.save_checkpoint_manager(1)
    step = 0

    batches_per_epoch = len(dataloader_train)
    total_train_batches = args.num_epochs * batches_per_epoch
    completed_train_batches = 0
    training_start_time = time.time()
    validation_durations = []
    total_validation_runs = sum(
        1 for epoch_index in range(args.num_epochs)
        if (epoch_index + 1) % args.val_step == 0
    )

    def estimate_total_remaining_time():
        if completed_train_batches == 0:
            return None

        elapsed_wall_time = time.time() - training_start_time
        observed_validation_time = sum(validation_durations)
        observed_train_time = max(elapsed_wall_time - observed_validation_time, 0.0)
        average_train_batch_time = observed_train_time / completed_train_batches
        remaining_train_batches = total_train_batches - completed_train_batches

        average_validation_time = (
            observed_validation_time / len(validation_durations)
            if validation_durations else 0.0
        )
        remaining_validation_runs = total_validation_runs - len(validation_durations)

        return (
            average_train_batch_time * remaining_train_batches
            + average_validation_time * remaining_validation_runs
        )

    for epoch in range(args.num_epochs):
        model.train()
        lr = utils.poly_learning_rate(args, opt, epoch) 
        lr = opt.param_groups[-1]['lr']
        with tqdm(total=num_train_data, desc=f'[Train] fold[{num_fold}/{args.k_fold}] Epoch[{epoch + 1}/{args.num_epochs} LR={lr:.8f}] ', unit='img') as pbar:
            for batch in dataloader_train:
                step += 1
                image = batch["image"]
                label = batch["label"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3

                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                opt.zero_grad()
                
                outputs = normalize_outputs(model(image))
                main_out = outputs["main_out"]

                diceloss = criterion_dice(main_out, label)
                celoss = criterion(main_out, label)
                totall_loss = celoss

                if "aux_out" in outputs.keys():
                    aux_losses = 0
                    for aux in outputs["aux_out"]:
                        label_aux = label
                        auxloss = (criterion_dice(aux, label)) *  args.aux_weight 
                        totall_loss += auxloss
                        aux_losses += auxloss

                totall_loss.backward()
                opt.step()

                completed_train_batches += 1
                total_eta_seconds = estimate_total_remaining_time()
                total_progress = 100.0 * completed_train_batches / total_train_batches

                if step % 5 == 0:
                    if args.aux:
                        writer.add_scalar("Train/aux_losses",aux_losses, step)
                    writer.add_scalar("Train/Totall_loss", totall_loss.item(), step)
                    writer.add_scalar("Train/lr", lr, step)

                pbar.set_postfix(**{
                    'loss': f'{totall_loss.item():.4f}',
                    'total_progress': f'{total_progress:.1f}%',
                    'total_eta': format_duration(total_eta_seconds),
                })
                pbar.update(args.batch_size)
                
        if (epoch+1) % args.val_step == 0:
            validation_start_time = time.time()
            mDice, mIoU, mAcc, mSensitivity, mSpecificity, mPrecision, mAuc, mBACC = val(model, dataloader_val, num_train_val, device, args)
            validation_durations.append(time.time() - validation_start_time)
            writer.add_scalar("Valid/Dice_val", mDice, step)
            writer.add_scalar("Valid/IoU_val", mIoU, step)
            writer.add_scalar("Valid/Acc_val", mAcc, step)
            writer.add_scalar("Valid/Precision_val", mPrecision, step)
            writer.add_scalar("Valid/Auc_val", mAuc, step)
            writer.add_scalar("Valid/Sen_val", mSensitivity, step)
            writer.add_scalar("Valid/Spe_val", mSpecificity, step)
            writer.add_scalar("Valid/bacc_val", mBACC, step)
            val_result = [num_fold, epoch+1, mDice, mIoU, mAcc, mPrecision, mAuc, mSensitivity, mSpecificity, mBACC]
            with open(args.val_result_file, "a") as f:
                w = csv.writer(f)
                w.writerow(val_result)
            cp_manager.save(model, opt, os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{epoch + 1}.pth'), float(mDice))

            total_eta_seconds = estimate_total_remaining_time()
            elapsed_training_time = time.time() - training_start_time
            print(
                f'[Overall] Epoch {epoch + 1}/{args.num_epochs}, '
                f'elapsed={format_duration(elapsed_training_time)}, '
                f'total_eta={format_duration(total_eta_seconds)}'
            )


def val(model, dataloader, num_train_val,  device, args):
    all_dice = []
    all_iou = []
    all_acc = []
    all_precision = []
    all_auc = []
    all_sen = []
    all_spe = []
    all_bacc = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=num_train_val, desc=f'VAL', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                file = batch["file"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                outputs = normalize_outputs(model(image))
                main_out = outputs["main_out"]
                main_out = torch.exp(main_out).max(dim=1)[1] 

                for b in range(image.size()[0]):
                    file_name, _ = os.path.splitext(file[b])
                    hist = utils.fast_hist(label[b, :, :], main_out[b, :, :], args.n_class)
                    dice, iou, acc, Precision, Sensitivity, Specificity, BACC = utils.cal_scores(hist.cpu().numpy())
                    auc = utils.calc_auc(main_out[b, :, :], label[b, :, :])
                    all_dice.append(list(dice))
                    all_iou.append(list(iou))
                    all_acc.append([acc])
                    all_precision.append([Precision] if np.isscalar(Precision) else list(Precision))
                    all_auc.append([auc])
                    all_sen.append(list(Sensitivity))
                    all_spe.append(list(Specificity))
                    all_bacc.append(list(BACC))
                pbar.update(image.size()[0])
    mDice = np.array(all_dice).mean()
    mIoU = np.array(all_iou).mean()
    mAcc = np.array(all_acc).mean()
    mPrecision = np.array(all_precision).mean()
    mAuc = np.array(all_auc).mean()
    mSensitivity = np.array(all_sen).mean()
    mSpecificity = np.array(all_spe).mean()
    mBACC = np.array(all_bacc).mean()
    
    print(f'\r   [VAL] mDice:{mDice:0.2f}, mIoU:{mIoU:0.2f}, mAcc:{mAcc:0.2f}, mPrecision:{mPrecision:0.2f}, mAuc:{mAuc:0.2f},  mSen:{mSensitivity:0.2f}, mSpec:{mSpecificity:0.2f}, mBACC:{mBACC:0.2f}')

    return mDice, mIoU, mAcc, mSensitivity, mSpecificity, mPrecision, mAuc, mBACC



def test(model, device, args, num_fold=0):
    if os.path.exists(args.val_result_file):
        with open(args.val_result_file, "r") as f:
            reader = csv.reader(f)
            val_result = list(reader)
        best_epoch = utils.best_model_in_fold(val_result, num_fold)
    else:
        best_epoch = args.num_epochs
    model_dir = os.path.join(args.checkpoint_dir[num_fold], f'CP_epoch{best_epoch}.pth')
    model.load_state_dict(torch.load(model_dir, map_location=device)["state_dict"])
    print(f'\rtest model loaded: [fold:{num_fold}] [best_epoch:{best_epoch}]')

    dataset_test = myDataset(args.data_root, args.target_root, args.crop_size, "test",
                                k_fold=args.k_fold, imagefile_csv=args.dataset_file_list, num_fold=num_fold,data_root_aux=args.data_root_aux,)
    dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    all_dice = []
    all_iou = []
    all_acc = []
    all_precision = []
    all_auc = []
    all_sen = []
    all_spe = []
    all_bacc = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset_test), desc=f'TEST fold {num_fold}/{args.k_fold}', unit='img') as pbar:
            for batch in dataloader:
                image = batch["image"]
                label = batch["label"]
                file = batch["file"]
                assert len(image.size()) == 4
                assert len(label.size()) == 3
                image = image.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                outputs = normalize_outputs(model(image))
                pred = outputs["main_out"]

                if args.tt_aug:
                    for i, axis in enumerate([[2], [3], [2, 3]]):
                        image_tmp = torch.flip(image, dims=axis)
                        pred_tmp = normalize_outputs(model(image_tmp))["main_out"]
                        pred_tmp = torch.flip(pred_tmp, dims=axis)
                        pred += pred_tmp
                    pred = pred / 4

                pred = torch.exp(pred).max(dim=1)[1]

                for b in range(image.size()[0]):
                    hist = utils.fast_hist(label[b,:,:], pred[b,:,:], args.n_class)
                    dice, iou, acc, Precision, Sensitivity, Specificity, bacc = utils.cal_scores(hist.cpu().numpy(), smooth=0.01)
                    auc = utils.calc_auc(pred[b, :, :], label[b, :, :])

                    test_result = [file[b], dice.mean()]+list(dice)+[iou.mean()]+list(iou)+[acc] + [Precision.mean()] + [auc] + \
                        [Sensitivity.mean()]+list(Sensitivity)+[Specificity.mean()]+list(Specificity)+ \
                        [bacc.mean()]+list(bacc)
                    with open(args.test_result_file, "a") as f:
                        w = csv.writer(f)
                        w.writerow(test_result)

                    all_dice.append(list(dice))
                    all_iou.append(list(iou))
                    all_acc.append([acc])
                    all_precision.append([Precision.mean()])
                    all_auc.append([auc])
                    all_sen.append(list(Sensitivity))
                    all_spe.append(list(Specificity))
                    all_bacc.append(list(bacc))
                    if args.plot:
                        file_name, _ = os.path.splitext(file[b])
                        save_image(pred[b,:,:].cpu().float().unsqueeze(0), os.path.join(args.plot_save_dir, file_name + f"_pred_{dice.mean():.2f}.png"), normalize=True)

                pbar.update(image.size()[0])

    print(f"\r---------Fold {num_fold} Test Result---------")
    print(f'mDice: {np.array(all_dice).mean()}')
    print(f'mIoU:  {np.array(all_iou).mean()}')
    print(f'mAcc:  {np.array(all_acc).mean()}')
    print(f'mPrecision: {np.array(all_precision).mean()}')
    print(f'mAuc:  {np.array(all_auc).mean()}')
    print(f'mSens: {np.array(all_sen).mean()}')
    print(f'mSpec: {np.array(all_spe).mean()}')
    print(f'mBACC: {np.array(all_bacc).mean()}')

    if num_fold == 0:
        utils.save_print_score(all_dice, all_iou, all_acc, all_precision, all_auc, all_sen, all_spe, all_bacc, args.test_result_file, args.label_names)
        return

    return all_dice, all_iou, all_acc, all_precision, all_auc, all_sen, all_spe, all_bacc



if __name__ == "__main__":

    seed = int(os.environ.get("TRAIN_SEED", "12345"))
    print(f"Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    cudnn.deterministic = True
    cudnn.benchmark = False

    args = basic_setting()
    assert args.k_fold != 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    if (not os.path.exists(args.dataset_file_list)) and (args.k_fold is not None):
        utils.get_dataset_filelist(args.data_root, args.dataset_file_list)

    if args.dataset == 'OCTA-3M':
        from dataset.dataset_3M import *
    elif args.dataset == 'OCTA-6M':
        from dataset.dataset_6M import *
    elif args.dataset == 'ROSE':
        from dataset.dataset_ROSE import *
    elif args.dataset == 'ROSSA':
        from dataset.dataset_ROSSA import *
    elif args.dataset == 'OCTA-SS':
        from dataset.dataset_SS import *
    else:
        print("dataset is not chosen")

    mode = args.mode
    if args.k_fold is None:
        print("k_fold is None")
        if mode == "train_test":
            args.mode = "train"
            print("###################### Train Start & " +  args.network  + " ######################")
            main(args)
            args.mode = "test"
            print("###################### Test Start & " +  args.network  + " ######################")
            main(args)
        else:
            main(args)
    else:
        if mode == "train_test":
            print("###################### Train & Test Start & "+  args.network  + " ######################")

        if mode == "train" or mode == "train_test":
            args.mode = "train"
            print("###################### Train Start & " +  args.network  + " ######################")
            for i in range(args.start_fold, args.end_fold):
                torch.cuda.empty_cache()
                main(args, num_fold=i + 1)

        if mode == "test" or mode == "train_test":
            args.mode = "test"
            print("###################### Test Start & " + args.network + " ######################")
            all_dice = []
            all_iou = []
            all_acc = []
            all_precision = []
            all_sen = []
            all_spe = []
            for i in range(args.start_fold, args.end_fold):
                Dice, IoU, Acc, Precision, Sensitivity, Specificity = main(args, num_fold=i + 1)
                all_dice += Dice
                all_iou += IoU
                all_acc += Acc
                all_precision += Precision
                all_sen += Sensitivity
                all_spe += Specificity
            utils.save_print_score(all_dice, all_iou, all_acc, all_precision, [], all_sen, all_spe, [], args.test_result_file, args.label_names)




