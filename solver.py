import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim as optim
from networks1 import *
from measure import *
from loader import *


class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.REDCNN = RED_CNN()
        # self.feature_extractor = WGAN_VGG_FeatureExtractor()
        # self.U_Net = U_Net1()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
            # self.U_Net = nn.DataParallel(self.U_Net)
            # self.feature_extractor = nn.DataParallel(self.feature_extractor)
        self.REDCNN.to(self.device)
        # self.feature_extractor.to(self.device)
        # self.U_Net.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.criterion1 = nn.L1Loss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)
        # self.optimizer1 = optim.Adam(self.feature_extractor.parameters(), self.lr)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # for param_group in self.optimizer1.param_groups:
        #     param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 4, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2],
                                                                           original_result[3]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2],
                                                                           pred_result[3]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self):
        train_losses = []
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        total_iters = 0
        param = self.get_parameter_number(self.REDCNN)
        print('param=', param)
        for epoch in range(0, self.num_epochs):
            self.REDCNN.train(True)
            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1
                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                if self.patch_size:  # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)
                pred, _ = self.REDCNN(x)
                loss = []
                # self.p_criterion = nn.L1Loss()
                for i in range(len(pred)):
                    # fake_feature = self.feature_extractor(pred[i].repeat(1, 3, 1, 1))
                    # real_feature = self.feature_extractor(y.repeat(1, 3, 1, 1))
                    # # loss1 = self.criterion(pred[i], y)+((0.6/(i+1))*self.criterion1(pred[i], y))
                    # loss1 = self.p_criterion(fake_feature, real_feature)
                    loss2 = self.criterion(pred[i], x - y)  # + 0.1 * self.criterion1(pred[i], x - y)       # + ((0.6*(i + 1)) * self.criterion1(pred[i], x - y))  # 0.4
                    # loss1 = self.criterion(pred[i], y) + (0.1 * self.criterion1(pred[i], y))
                    # loss2 += 0.1*loss1
                    loss.append(loss2)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()
                for i1 in range(len(loss)):
                    # self.optimizer1.zero_grad()
                    loss[i1].backward(retain_graph=True)
                self.optimizer.step()
                    # self.optimizer1.step()


                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS1: {:.8f}, LOSS2: {:.8f}, LOSS3: {:.8f}, LOSS4: {:.8f}, LOSS5: {:.8f}".format(total_iters, epoch,
                                                                                                            self.num_epochs, iter_+1,
                                                                                                            len(self.data_loader), loss[0].item(),
                                                                                                            loss[1].item(),loss[2].item(),loss[3].item(),loss[4].item()))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))

            #???????????????????????????????????????????????????????????????????????????????????????????????????????????????

            if total_iters % self.save_iters == 0 and total_iters != 0:    # 修改
                dataset_ = ct_dataset(mode='test', load_mode=0, saved_path='G:/深度学习/级联去噪/test(2)/',
                                      test_patient='LDCT', patch_n=None,
                                      patch_size=None, transform=False)
                data_loader = DataLoader(dataset=dataset_, batch_size=None, shuffle=True, num_workers=7)

                self.WGAN_VGG_generator1 = RED_CNN().to(self.device)
                f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format((epoch + 1) * self.save_iters))  # 修改
                if self.multi_gpu:
                    state_d = OrderedDict()
                    for k, v in torch.load(f):
                        n = k[7:]
                        state_d[n] = v
                    self.WGAN_VGG_generator1.load_state_dict(state_d)
                else:
                    self.WGAN_VGG_generator1.load_state_dict(torch.load(f))

                with torch.no_grad():

                    for i, (x, y) in enumerate(data_loader):
                        # (x, y) = list(self.data_loader)[i]

                        shape_ = x.shape[-1]
                        x = x.unsqueeze(0).float().to(self.device)
                        y = y.unsqueeze(0).float().to(self.device)
                        x = x.unsqueeze(0).float().to(self.device)
                        y = y.unsqueeze(0).float().to(self.device)
                        _, pred = self.WGAN_VGG_generator1(x)
                        x1 = self.trunc(self.denormalize_(x))
                        y1 = self.trunc(self.denormalize_(y))
                        pred1 = self.trunc(self.denormalize_(pred[4]))
                        data_range = self.trunc_max - self.trunc_min
                        original_result, pred_result = compute_measure(x1, y1, pred1, data_range)
                        # ori_psnr_avg = original_result[0]
                        # ori_ssim_avg = original_result[1]
                        # ori_rmse_avg = original_result[2]
                        pred_psnr_avg += pred_result[0]
                        pred_ssim_avg += pred_result[1]
                        pred_rmse_avg += pred_result[2]



            #########################################################
            # 日志文件
            # with open('Loss.txt', 'a') as f:
            #     f.write('ITER:%d loss:%.20f' % (total_iters, loss) + '\n')
            #     f.close()

                with open('./save/pred_psnr_avg.txt', 'a') as f:
                    f.write('EPOCH:%d loss:%.20f' % (epoch, pred_psnr_avg / len(data_loader)) + '\n')
                    f.close()

                with open('./save/pred_ssim_avg.txt', 'a') as f:
                    f.write('EPOCH:%d loss:%.20f' % (epoch, pred_ssim_avg / len(data_loader)) + '\n')
                    f.close()

                with open('./save/pred_rmse_avg.txt', 'a') as f:
                    f.write('EPOCH:%d loss:%.20f' % (epoch, pred_rmse_avg / len(data_loader)) + '\n')
                    f.close()
                pred_psnr_avg = 0
                pred_ssim_avg = 0
                pred_rmse_avg = 0
            #########################################################
            else:
                continue

    def test(self):
        del self.REDCNN
        # load
        self.REDCNN = RED_CNN().to(self.device)
        self.load_model(self.test_iters)
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg2, pred_ssim_avg2, pred_rmse_avg2 = 0, 0, 0
        pred_psnr_avg3, pred_ssim_avg3, pred_rmse_avg3 = 0, 0, 0
        pred_psnr_avg4, pred_ssim_avg4, pred_rmse_avg4 = 0, 0, 0
        pred_psnr_avg5, pred_ssim_avg5, pred_rmse_avg5 = 0, 0, 0
        pred_psnr_avg6, pred_ssim_avg6, pred_rmse_avg6 = 0, 0, 0
        pred_psnr, pred_ssim, pred_rmse = [], [], []
        # self.REDCNN = U_Net().to(self.device)

        x_path = os.path.join(self.save_path, 'x')
        if not os.path.exists(x_path):
            os.makedirs(x_path)
            print('Create path : {}'.format(x_path))
        y_path = os.path.join(self.save_path, 'y')
        if not os.path.exists(y_path):
            os.makedirs(y_path)
            print('Create path : {}'.format(y_path))
        pred1_path = os.path.join(self.save_path, 'pred1')
        if not os.path.exists(pred1_path):
            os.makedirs(pred1_path)
            print('Create path : {}'.format(pred1_path))
        pred2_path = os.path.join(self.save_path, 'pred2')
        if not os.path.exists(pred2_path):
            os.makedirs(pred2_path)
            print('Create path : {}'.format(pred2_path))
        pred3_path = os.path.join(self.save_path, 'pred3')
        if not os.path.exists(pred3_path):
            os.makedirs(pred3_path)
            print('Create path : {}'.format(pred3_path))
        pred4_path = os.path.join(self.save_path, 'pred4')
        if not os.path.exists(pred4_path):
            os.makedirs(pred4_path)
            print('Create path : {}'.format(pred4_path))
        pred5_path = os.path.join(self.save_path, 'pred5')
        if not os.path.exists(pred5_path):
            os.makedirs(pred5_path)
            print('Create path : {}'.format(pred5_path))
        with torch.no_grad():
            for i, (x1, y1) in enumerate(self.data_loader):
                shape_ = x1.shape[-1]
                x1 = x1.unsqueeze(0).float().to(self.device)
                y1 = y1.unsqueeze(0).float().to(self.device)
                _, pred11 = self.REDCNN(x1)
                plt_data = []
                plt_data_pred = []
                psnr_result = []
                ssim_result = []
                rmse_result = []
                x1 = self.trunc(self.denormalize_(x1.view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'x', '{}_result'.format(i)), x1)
                pred1111 = self.trunc(self.denormalize_(pred11[0].view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'pred1', '{}_result'.format(i)), pred1111)
                pred2222 = self.trunc(self.denormalize_(pred11[1].view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'pred2', '{}_result'.format(i)), pred2222)
                pred3333 = self.trunc(self.denormalize_(pred11[2].view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'pred3', '{}_result'.format(i)), pred3333)
                pred4444 = self.trunc(self.denormalize_(pred11[3].view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'pred4', '{}_result'.format(i)), pred4444)
                pred5555 = self.trunc(self.denormalize_(pred11[4].view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'pred5', '{}_result'.format(i)), pred5555)
                plt_data_pred.append(x1)
                for img in range(len(pred11)):
                    pred1 = self.trunc(self.denormalize_(pred11[img].view(shape_, shape_).cpu().detach()))
                    plt_data.append(pred1)
                    plt_data_pred.append(pred1)
                y1 = self.trunc(self.denormalize_(y1.view(shape_, shape_).cpu().detach()))
                np.save(os.path.join(self.save_path, 'y', '{}_result'.format(i)), y1)
                plt_data_pred.append(y1)
                data_range = self.trunc_max - self.trunc_min
                psnr_result1 = compute_measure1(x1, y1, data_range)
                ori_psnr_avg += psnr_result1
                psnr_result.append(psnr_result1)
                ssim_result1 = compute_measure2(x1, y1, data_range)
                ori_ssim_avg += ssim_result1
                ssim_result.append(ssim_result1)
                rmse_result1 = compute_measure3(x1, y1)
                ori_rmse_avg += rmse_result1
                rmse_result.append(rmse_result1)
                pred_psnr1 = []
                pred_ssim1 = []
                pred_rmse1 = []
                for i2 in range(len(plt_data)):
                    psnr_result2 = compute_measure1(plt_data[i2], y1, data_range)
                    pred_psnr1.append(psnr_result2)
                    psnr_result.append(psnr_result2)
                    ssim_result2 = compute_measure2(plt_data[i2], y1, data_range)
                    pred_ssim1.append(ssim_result2)
                    ssim_result.append(ssim_result2)
                    rmse_result2 = compute_measure3(plt_data[i2], y1)
                    pred_rmse1.append(rmse_result2)
                    rmse_result.append(rmse_result2)
                psnr_result.append(psnr_result1)
                ssim_result.append(ssim_result1)
                rmse_result.append(rmse_result1)
                pred_psnr.append(pred_psnr1)
                pred_ssim.append(pred_ssim1)
                pred_rmse.append(pred_rmse1)

                ## figure
                fig_titles = ['LDCT', 'D = 1', 'D = 2', 'D=3', 'D=4', 'D=5', 'NDCT']
                plt.figure()
                f, axs = plt.subplots(1, 7, figsize=(30, 10))
                for i1, img1 in enumerate(plt_data_pred):
                    axs[i1].imshow(img1, cmap=plt.cm.gray)
                    axs[i1].set_title(fig_titles[i1], fontsize=30)
                    axs[i1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(psnr_result[i1],
                                                                                         ssim_result[i1],
                                                                                         rmse_result[i1]), fontsize=20)

                f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(i)))
                plt.close()
            for i5 in range(np.array(pred_psnr).shape[0]):
                pred_psnr_avg2 += pred_psnr[i5][0]
                pred_psnr_avg3 += pred_psnr[i5][1]
                pred_psnr_avg4 += pred_psnr[i5][2]
                pred_psnr_avg5 += pred_psnr[i5][3]
                pred_psnr_avg6 += pred_psnr[i5][4]
                pred_ssim_avg2 += pred_ssim[i5][0]
                pred_ssim_avg3 += pred_ssim[i5][1]
                pred_ssim_avg4 += pred_ssim[i5][2]
                pred_ssim_avg5 += pred_ssim[i5][3]
                pred_ssim_avg6 += pred_ssim[i5][4]
                pred_rmse_avg2 += pred_rmse[i5][0]
                pred_rmse_avg3 += pred_rmse[i5][1]
                pred_rmse_avg4 += pred_rmse[i5][2]
                pred_rmse_avg5 += pred_rmse[i5][3]
                pred_rmse_avg6 += pred_rmse[i5][4]


            print('\n')
            print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.data_loader),
                ori_ssim_avg / len(self.data_loader),
                ori_rmse_avg / len(self.data_loader)))
            print('After learning\nPSNR1 avg: {:.4f} \nSSIM1 avg: {:.4f} \nRMSE1 avg: {:.4f}'.format(
                pred_psnr_avg2 / len(self.data_loader),
                pred_ssim_avg2 / len(self.data_loader),
                pred_rmse_avg2 / len(self.data_loader)))
            print('After learning\nPSNR2 avg: {:.4f} \nSSIM2 avg: {:.4f} \nRMSE2 avg: {:.4f}'.format(
                pred_psnr_avg3 / len(self.data_loader),
                pred_ssim_avg3 / len(self.data_loader),
                pred_rmse_avg3 / len(self.data_loader)))
            print('After learning\nPSNR3 avg: {:.4f} \nSSIM3 avg: {:.4f} \nRMSE3 avg: {:.4f}'.format(
                pred_psnr_avg4 / len(self.data_loader),
                pred_ssim_avg4 / len(self.data_loader),
                pred_rmse_avg4 / len(self.data_loader)))
            print('After learning\nPSNR4 avg: {:.4f} \nSSIM4 avg: {:.4f} \nRMSE4 avg: {:.4f}'.format(
                pred_psnr_avg5 / len(self.data_loader),
                pred_ssim_avg5 / len(self.data_loader),
                pred_rmse_avg5 / len(self.data_loader)))
            print('After learning\nPSNR5 avg: {:.4f} \nSSIM5 avg: {:.4f} \nRMSE5 avg: {:.4f}'.format(
                pred_psnr_avg6 / len(self.data_loader),
                pred_ssim_avg6 / len(self.data_loader),
                pred_rmse_avg6 / len(self.data_loader)))
