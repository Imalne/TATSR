import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy
from utils import util, ssim_psnr
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
from thop import profile
from PIL import Image
import numpy as np

sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.meters import AverageMeter
from utils.metrics import get_str_list, Accuracy
from utils.util import str_filt
from utils import utils_moran



class TextSR(base.TextBase):
    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        aster, aster_info = self.Aster_init()
        optimizer_G = self.optimizer_init(model)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        for epoch in range(cfg.epochs):
            self.update_lr(optimizer_G, epoch)
            for j, data in (enumerate(train_loader)):
                model.train()

                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j + 1

                if cfg.dataset.heat_map:
                    images_hr, images_lr, label_strs, images_mask = data
                else:
                    images_hr, images_lr, label_strs = data
                    images_mask = None

                if self.args.syn:
                    images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                      self.config.TRAIN.width // self.scale_factor),
                                                          mode='bicubic')
                    images_lr = images_lr.to(self.device)
                else:
                    images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                if cfg.dataset.heat_map:
                    images_mask = images_mask.to(self.device)

                image_sr = model(images_lr)

                loss_im, multi_loss = image_crit(image_sr, images_hr, weight = images_mask)
                loss_im = loss_im.mean()* 100


                optimizer_G.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                # print(torch.mean(torch.abs(model.module.block2.conv1.weight.grad)))
                optimizer_G.step()



                # torch.cuda.empty_cache()
                if iters % cfg.displayInterval == 0:
                    print('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          '{:.3f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data)))
                self.save_train_log(float(loss_im.data), iters, multi_loss)

                if iters % cfg.VAL.valInterval == 0:
                    print('======================================================')
                    current_acc_dict = {}
                    dataset_size={}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        dataset_size[data_name] = len(val_loader.dataset)
                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                        self.save_valid_log(converge_list[-1], data_name)
                    if sum(dataset_size[k] * v for k, v in current_acc_dict.items()) > best_acc:
                        best_acc = sum(dataset_size[k] * v for k, v in current_acc_dict.items())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list)


    def eval(self, model, val_loader, image_crit, index, aster, aster_info):
        for p in model.parameters():
            p.requires_grad = False
        for p in aster.parameters():
            p.requires_grad = False
        model.eval()
        aster.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            with torch.no_grad():
                images_sr = model(images_lr)
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))
            aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
            aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
            with torch.no_grad():
                aster_output_lr = aster(aster_dict_lr)
                aster_output_sr = aster(aster_dict_sr)
            pred_rec_lr = aster_output_lr['output']['pred_rec']
            pred_rec_sr = aster_output_sr['output']['pred_rec']
            pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
            for pred, target in zip(pred_str_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1

            loss_im, _= image_crit(images_sr, images_hr)
            loss_im = loss_im.mean()
            loss_rec = aster_output_sr['losses']['loss_rec'].mean()
            sum_images += val_batch_size
            # torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        print('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(loss_rec.data), 0,
                      float(psnr_avg), float(ssim_avg), ))
        print('save display images')
        self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        print('aster_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg
        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        # print(sum(p.numel() for p in moran.parameters()))
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        sr_time = 0
        
        psnr_sr_list=[]
        psnr_lr_list=[]
        ssim_sr_list=[]
        ssim_lr_list=[]
        image_sr_list=[]
        image_hr_list=[]
        image_lr_list=[]
        label_pred_lr_list=[]
        label_pred_sr_list=[]
        label_pred_hr_list=[]
        label_target_list=[]
        
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            sr_beigin = time.time()
            with torch.no_grad():
                images_sr = model(images_lr)
            print(i)
            
            image_lr_list.append(images_lr[:,:3,:,:].cpu())
            image_sr_list.append(images_sr[:,:3,:,:].cpu())
            image_hr_list.append(images_hr[:,:3,:,:].cpu())
            label_target_list.append(label_strs[0])
            upsampled_images_lr = nn.functional.interpolate(images_lr, scale_factor=2, mode='bicubic')

            # images_sr = images_lr
            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))
            
            psnr_sr_list.append(self.cal_psnr(images_sr, images_hr).cpu())
            psnr_lr_list.append(self.cal_psnr(upsampled_images_lr, images_hr).cpu())
            ssim_sr_list.append(self.cal_ssim(images_sr, images_hr).cpu())
            ssim_lr_list.append(self.cal_ssim(upsampled_images_lr, images_hr).cpu())
            

            if self.args.rec == 'moran':


                moran_input = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds]
                label_pred_lr_list.append(pred_str_lr)


                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                label_pred_sr_list.append(pred_str_sr)

                moran_input = self.parse_moran_data(images_hr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_hr = [pred.split('$')[0] for pred in sim_preds]
                label_pred_hr_list.append(pred_str_hr)

                

            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                with torch.no_grad():
                    aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
                label_pred_sr_list.append(pred_str_sr)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                with torch.no_grad():
                    aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                label_pred_lr_list.append(pred_str_lr)

                aster_dict_lr = self.parse_aster_data(images_hr[:, :3, :, :])
                with torch.no_grad():
                    aster_output_lr = aster(aster_dict_lr)
                pred_rec_hr = aster_output_lr['output']['pred_rec']
                pred_str_hr, _ = get_str_list(pred_rec_hr, aster_dict_lr['rec_targets'], dataset=aster_info)
                label_pred_hr_list.append(pred_str_hr)


            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                label_pred_sr_list.append(pred_str_sr)

                crnn_input = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                label_pred_lr_list.append(pred_str_lr)


                crnn_input = self.parse_crnn_data(images_hr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_hr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                label_pred_hr_list.append(pred_str_hr)
                
            
            
            for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                    n_correct += 1
            sum_images += val_batch_size
            # torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))
            # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
            
            
        
        
        os.makedirs(self.args.save_dir, exist_ok=True)
        np.save(os.path.join(self.args.save_dir,'image_hr.npy'), torch.cat(image_hr_list, dim=0).numpy())
        np.save(os.path.join(self.args.save_dir,'image_sr.npy'), torch.cat(image_sr_list, dim=0).numpy())
        np.save(os.path.join(self.args.save_dir,'image_lr.npy'), torch.cat(image_lr_list, dim=0).numpy())
        np.save(os.path.join(self.args.save_dir,'psnr_sr.npy'), np.array(psnr_sr_list))
        np.save(os.path.join(self.args.save_dir,'ssim_sr.npy'), np.array(ssim_sr_list))
        np.save(os.path.join(self.args.save_dir,'psnr_lr.npy'), np.array(psnr_lr_list))
        np.save(os.path.join(self.args.save_dir,'ssim_lr.npy'), np.array(ssim_lr_list))
        np.save(os.path.join(self.args.save_dir,'label_pred_lr.npy'), np.array(label_pred_lr_list))
        np.save(os.path.join(self.args.save_dir,'label_pred_sr.npy'), np.array(label_pred_sr_list))
        np.save(os.path.join(self.args.save_dir,'label_pred_hr.npy'), np.array(label_pred_hr_list))
        np.save(os.path.join(self.args.save_dir,'label_target.npy'), np.array(label_target_list))
        
        
        
        
        
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                with torch.no_grad():
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                with torch.no_grad():
                    moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                     debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                with torch.no_grad():
                    aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                with torch.no_grad():
                    aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                with torch.no_grad():
                    crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                with torch.no_grad():
                    crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            print(pred_str_lr, '===>', pred_str_sr)
            # torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)


if __name__ == '__main__':
    embed()
