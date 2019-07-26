import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap, eval_market1501
from utils.re_ranking import re_ranking

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, (inputs, labels, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')

        self.loss.end_log(len(self.train_loader))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 5))
        qf, q_image_paths = self.extract_feature(self.query_loader)
        gf, g_image_paths = self.extract_feature(self.test_loader)
        qf = qf.numpy()  # (31, 2048) for debug
        gf = gf.numpy()  # (154, 2048) for

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf, metric='euclidean')

        if self.args.multi_query:
            # Suitable for boxes dataset only
            # Use several same id query samples to retrieval one gallery
            print(dist.shape)  # (31, 154)
            ids = [int(x.split(os.sep)[-1].split('_')[0]) for x in q_image_paths]
            ids_set = set(ids)
            ids_np = np.asarray(ids, dtype=np.int32)

            for id in ids_set:
                indices = np.where(ids_np == id)[0]
                mean = np.mean(dist[indices, :], axis=0)
                for index in indices:
                    dist[index, :] = mean

        if self.args.multi_gallery:
            # Suitable for boxes dataset only
            # query samples to retrieval images and mean the same id of gallery
            print(dist.shape)  # (31, 154) for debug

            ids = [int(x.split(os.sep)[-1].split('_')[0]) for x in g_image_paths]

            ids_set = set(ids)
            ids_np = np.asarray(ids, dtype=np.int32)

            for i, q_image_path in enumerate(q_image_paths):
                for id in ids_set:
                    indices = np.where(ids_np == id)[0]
                    mean = np.mean(dist[i, indices], axis=0)
                    dist[i, indices] = mean

        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
            best[0][0],
            (best[1][0] + 1)*self.args.test_every
            )
        )
        # Another method to calculate cmc and mAP, it get the same result!
        # print(eval_market1501(dist, self.queryset.ids, self.testset.ids,
        #                       self.queryset.cameras, self.testset.cameras, 100))

        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))

    def retrieval(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 5))
        qf, q_image_paths = self.extract_feature(self.query_loader)
        gf, g_image_paths = self.extract_feature(self.test_loader)
        qf = qf.numpy()  # (31, 2048) for debug
        gf = gf.numpy()  # (154, 2048) for debug

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)  # (31, 154) for debug

        if self.args.multi_query:
            # Suitable for boxes dataset only
            # Use several same id query samples to retrieval one gallery
            print(dist.shape)  # (31, 154) for debug
            ids = [int(x.split(os.sep)[-1].split('_')[0]) for x in q_image_paths]
            ids_set = set(ids)
            ids_np = np.asarray(ids, dtype=np.int32)

            for id in ids_set:
                indices = np.where(ids_np == id)[0]
                mean = np.mean(dist[indices, :], axis=0)
                for index in indices:
                    dist[index, :] = mean

        if self.args.multi_gallery:
            # Suitable for boxes dataset only
            # query samples to retrieval images and mean the same id of gallery
            print(dist.shape)  # (31, 154) for debug

            ids = [int(x.split(os.sep)[-1].split('_')[0]) for x in g_image_paths]

            ids_set = set(ids)
            ids_np = np.asarray(ids, dtype=np.int32)

            for i, q_image_path in enumerate(q_image_paths):
                for id in ids_set:
                    indices = np.where(ids_np == id)[0]
                    mean = np.mean(dist[i, indices], axis=0)
                    dist[i, indices] = mean

        # Output csv file
        indices = np.argsort(dist, axis=1)
        import csv
        with open(os.path.join('retrieval.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for i, q_image_path in enumerate(q_image_paths):
                row_string = [os.path.basename(q_image_path)]
                for j, index in enumerate(indices[i]):
                    row_string.append(os.path.basename(g_image_paths[index]))
                    row_string.append(str(-np.log(dist[i, index] + 1e-6)))
                spamwriter.writerow(row_string)

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        image_paths = list()
        for (inputs, labels, image_path) in loader:
            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
            for path in image_path:
                image_paths.append(path)
        return features, image_paths

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        elif self.args.retrieval_only:
            self.retrieval()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

