import os
import sys
import csv
from collections import defaultdict

import numpy as np
import scipy
import torch
from torch import optim
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network import AENet
from networks.criterion.mahala import loss_function_mahala, calc_inv_cov
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata


class DCASE2023T2AE(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(
            args=args,
            train=train,
            test=test
        )
        parameter_list = [{"params": self.model.parameters()}]
        self.optimizer = optim.AdamW(
            parameter_list,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        if self.optim_state_dict is not None:
            self.optimizer.load_state_dict(self.optim_state_dict)
        self.mse_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.mahala_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"
        self.latent_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_latent.pickle"
        self.hybrid_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_hybrid.pickle"

    def init_model(self):
        self.block_size = self.data.height
        return AENet(
            input_dim=self.data.input_dim,
            block_size=self.block_size,
            num_conditions=self.data.num_classes,
            latent_dim=self.args.latent_dim,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout,
        )

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["latent_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,latent_loss,recon_loss_source,recon_loss_target"

    def train(self, epoch):
        if epoch <= self.epoch:
            return
        if epoch == self.args.epochs + 1:
            self._fit_post_training_statistics()
            self._save_model(epoch=self.args.epochs)
            return

        self.model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_latent_loss = 0.0
        train_recon_loss_source = 0.0
        train_recon_loss_target = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            data, condition, _, machine_id, source_mask, target_mask = self._unpack_batch(batch)
            if data.shape[0] <= 1:
                continue

            self.optimizer.zero_grad()
            recon_batch, z = self.model(self._corrupt_input(data), condition)

            score_2d = self.train_loss_fn(recon_batch, data)
            score = self.loss_reduction_1d(score=score_2d)
            recon_loss = score.mean()
            latent_loss = self._latent_center_loss(z, machine_id, source_mask, target_mask)
            self.loss = recon_loss + self.args.latent_loss_weight * latent_loss
            self.loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            recon_loss_source = self._masked_mean(score, source_mask)
            recon_loss_target = self._masked_mean(score, target_mask)

            train_loss += float(self.loss.item())
            train_recon_loss += float(recon_loss.item())
            train_latent_loss += float(latent_loss.item())
            train_recon_loss_source += float(recon_loss_source.item())
            train_recon_loss_target += float(recon_loss_target.item())
            batch_count += 1

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    self.loss.item()))

        val_loss = self._validation_loss()
        batch_count = max(batch_count, 1)
        print('====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}'.format(
            epoch,
            train_loss / batch_count,
            val_loss))

        with open(self.log_path, 'a') as log:
            np.savetxt(log, ["{0},{1},{2},{3},{4},{5}".format(
                train_loss / batch_count,
                val_loss,
                train_recon_loss / batch_count,
                train_latent_loss / batch_count,
                train_recon_loss_source / batch_count,
                train_recon_loss_target / batch_count,
            )], fmt="%s")
        csv_to_figdata(
            file_path=self.log_path,
            column_heading_list=self.column_heading_list,
            ylabel="loss",
            fig_count=len(self.column_heading_list),
            cut_first_epoch=True
        )
        self._save_model(epoch=epoch)

    def _validation_loss(self):
        val_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            self.model.eval()
            for batch in self.valid_loader:
                data, condition, _, _, _, _ = self._unpack_batch(batch)
                recon_batch, _ = self.model(data, condition)
                loss = self.train_loss_fn(recon_batch, data).mean()
                val_loss += float(loss.item())
                batch_count += 1
        return val_loss / max(batch_count, 1)

    def _fit_post_training_statistics(self):
        print("\n============== FIT NORMAL STATISTICS ==============")
        self.model.eval()
        with torch.no_grad():
            self._fit_residual_and_latent_statistics()
            self._fit_score_distributions()

    def _fit_residual_and_latent_statistics(self):
        device = self.device
        num_classes = self.data.num_classes
        latent_dim = self.args.latent_dim
        eps = self.args.cov_eps

        residual_scatter_source = torch.zeros(self.block_size, self.block_size, device=device)
        residual_scatter_target = torch.zeros_like(residual_scatter_source)
        residual_count_source = 0
        residual_count_target = 0

        latent_sum_source = torch.zeros(num_classes, latent_dim, device=device)
        latent_sum_target = torch.zeros_like(latent_sum_source)
        latent_outer_source = torch.zeros(num_classes, latent_dim, latent_dim, device=device)
        latent_outer_target = torch.zeros_like(latent_outer_source)
        latent_count_source = torch.zeros(num_classes, device=device)
        latent_count_target = torch.zeros_like(latent_count_source)

        for loader in [self.train_loader, self.valid_loader]:
            for batch in tqdm(loader, desc="fit residual/latent"):
                data, condition, _, machine_id, source_mask, target_mask = self._unpack_batch(batch)
                recon_batch, z = self.model(data, condition)
                residual = data - recon_batch

                scatter, count = self._residual_scatter(residual, source_mask)
                residual_scatter_source += scatter
                residual_count_source += count
                scatter, count = self._residual_scatter(residual, target_mask)
                residual_scatter_target += scatter
                residual_count_target += count

                self._update_latent_accumulator(
                    z, machine_id, source_mask,
                    latent_sum_source, latent_outer_source, latent_count_source
                )
                self._update_latent_accumulator(
                    z, machine_id, target_mask,
                    latent_sum_target, latent_outer_target, latent_count_target
                )

        eye_residual = torch.eye(self.block_size, device=device)
        cov_source = self._finish_covariance(
            residual_scatter_source,
            residual_count_source,
            eye_residual,
            eps,
        )
        cov_target = self._finish_covariance(
            residual_scatter_target,
            residual_count_target,
            cov_source,
            eps,
        )
        self.model.cov_source.data = cov_source
        self.model.cov_target.data = cov_target

        eye_latent = torch.eye(latent_dim, device=device)
        mean_source, precision_source = self._finish_latent_bank(
            latent_sum_source,
            latent_outer_source,
            latent_count_source,
            eye_latent,
            eps,
        )
        mean_target, precision_target = self._finish_latent_bank(
            latent_sum_target,
            latent_outer_target,
            latent_count_target,
            eye_latent,
            eps,
            fallback_mean=mean_source,
            fallback_precision=precision_source,
        )
        self.model.latent_mean_source.data = mean_source
        self.model.latent_mean_target.data = mean_target
        self.model.latent_precision_source.data = precision_source
        self.model.latent_precision_target.data = precision_target
        self.model.latent_count_source.data = latent_count_source
        self.model.latent_count_target.data = latent_count_target

    def _fit_score_distributions(self):
        recon_scores, latent_scores = self._collect_raw_frame_scores()
        recon_scores = np.asarray(recon_scores, dtype=float)
        latent_scores = np.asarray(latent_scores, dtype=float)
        recon_mean = float(np.mean(recon_scores)) if len(recon_scores) else 0.0
        latent_mean = float(np.mean(latent_scores)) if len(latent_scores) else 0.0
        recon_std = float(np.std(recon_scores)) if len(recon_scores) else 1.0
        latent_std = float(np.std(latent_scores)) if len(latent_scores) else 1.0
        score_mean = torch.tensor([recon_mean, latent_mean], device=self.device)
        score_std = torch.tensor([
            max(recon_std, sys.float_info.epsilon),
            max(latent_std, sys.float_info.epsilon),
        ], device=self.device)
        self.model.score_mean.data = score_mean.float()
        self.model.score_std.data = score_std.float()

        score_dict = self._collect_file_score_dict()
        self.fit_anomaly_score_distribution(
            y_pred=score_dict["MSE"],
            score_distr_file_path=self.mse_score_distr_file_path
        )
        self.fit_anomaly_score_distribution(
            y_pred=score_dict["MAHALA"],
            score_distr_file_path=self.mahala_score_distr_file_path
        )
        self.fit_anomaly_score_distribution(
            y_pred=score_dict["LATENT"],
            score_distr_file_path=self.latent_score_distr_file_path
        )
        self.fit_anomaly_score_distribution(
            y_pred=score_dict["HYBRID"],
            score_distr_file_path=self.hybrid_score_distr_file_path
        )

    def _collect_raw_frame_scores(self):
        recon_scores = []
        latent_scores = []
        with torch.no_grad():
            for loader in [self.train_loader, self.valid_loader]:
                for batch in tqdm(loader, desc="collect raw scores"):
                    data, condition, _, _, _, _ = self._unpack_batch(batch)
                    recon_batch, z = self.model(data, condition)
                    recon_score = self.loss_reduction_1d(self.loss_fn(recon_batch, data))
                    latent_score = self._latent_frame_score(z, condition)
                    recon_scores.extend(recon_score.detach().cpu().numpy().tolist())
                    latent_scores.extend(latent_score.detach().cpu().numpy().tolist())
        return recon_scores, latent_scores

    def _collect_file_score_dict(self):
        inv_cov_source, inv_cov_target = calc_inv_cov(
            model=self.model,
            device=self.device
        )
        score_by_file = {
            "MSE": defaultdict(list),
            "MAHALA": defaultdict(list),
            "LATENT": defaultdict(list),
            "HYBRID": defaultdict(list),
        }
        with torch.no_grad():
            for loader in [self.train_loader, self.valid_loader]:
                for batch in tqdm(loader, desc="collect file scores"):
                    data, condition, names, _, _, _ = self._unpack_batch(batch)
                    recon_batch, z = self.model(data, condition)
                    frame_scores = self._all_frame_scores(
                        data=data,
                        condition=condition,
                        recon_data=recon_batch,
                        z=z,
                        inv_cov_source=inv_cov_source,
                        inv_cov_target=inv_cov_target,
                    )
                    for name_idx, basename in enumerate(names):
                        for score_name, score_values in frame_scores.items():
                            score_by_file[score_name][basename].append(float(score_values[name_idx].item()))

        return {
            score_name: [
                self._aggregate_score_values(file_scores)
                for file_scores in files.values()
            ]
            for score_name, files in score_by_file.items()
        }

    def _all_frame_scores(self, data, condition, recon_data, z, inv_cov_source, inv_cov_target):
        recon_score = self.loss_reduction_1d(self.loss_fn(recon_data, data))
        latent_score = self._latent_frame_score(z, condition)
        return {
            "MSE": recon_score,
            "MAHALA": self._mahala_frame_score(data, recon_data, inv_cov_source, inv_cov_target),
            "LATENT": latent_score,
            "HYBRID": self._hybrid_frame_score(recon_score, latent_score),
        }

    def _unpack_batch(self, batch):
        data = batch[0].to(self.device).float()
        condition = batch[2].to(self.device).float()
        names = list(batch[3])
        machine_id = torch.argmax(condition, dim=1).long()
        is_target = np.asarray(["target" in data_name for data_name in names], dtype=bool)
        target_mask = torch.from_numpy(is_target).to(self.device)
        source_mask = torch.logical_not(target_mask)
        return data, condition, names, machine_id, source_mask, target_mask

    def _corrupt_input(self, data):
        if self.args.input_noise <= 0:
            return data
        scale = data.detach().std(dim=1, keepdim=True).clamp_min(1.0)
        return data + torch.randn_like(data) * scale * self.args.input_noise

    def _latent_center_loss(self, z, machine_id, source_mask, target_mask):
        losses = []
        for domain_mask in [source_mask, target_mask]:
            if not torch.any(domain_mask):
                continue
            for class_id in torch.unique(machine_id[domain_mask]):
                mask = domain_mask & (machine_id == class_id)
                if int(mask.sum().item()) <= 1:
                    continue
                center = z[mask].mean(dim=0).detach()
                losses.append(torch.mean((z[mask] - center) ** 2))
        if len(losses) == 0:
            return z.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _residual_scatter(self, residual, mask):
        if not torch.any(mask):
            return torch.zeros(self.block_size, self.block_size, device=self.device), 0
        residual_frames = residual[mask].view(-1, self.block_size)
        count = residual_frames.size(0)
        if count <= 1:
            return torch.zeros(self.block_size, self.block_size, device=self.device), count
        centered = residual_frames - residual_frames.mean(dim=0, keepdim=True)
        return torch.matmul(centered.t(), centered), count

    def _update_latent_accumulator(self, z, machine_id, mask, latent_sum, latent_outer, latent_count):
        if not torch.any(mask):
            return
        for class_id in torch.unique(machine_id[mask]):
            class_mask = mask & (machine_id == class_id)
            class_z = z[class_mask]
            latent_sum[class_id] += class_z.sum(dim=0)
            latent_outer[class_id] += torch.matmul(class_z.t(), class_z)
            latent_count[class_id] += class_z.size(0)

    def _finish_covariance(self, scatter, count, fallback_cov, eps):
        if count > 1:
            cov = scatter / (count - 1)
            cov = 0.5 * (cov + cov.t())
            if torch.count_nonzero(cov).item() == 0:
                cov = fallback_cov.clone()
        else:
            cov = fallback_cov.clone()
        cov = cov + eps * torch.eye(cov.size(0), device=cov.device)
        return cov.float()

    def _finish_latent_bank(
        self,
        latent_sum,
        latent_outer,
        latent_count,
        default_precision,
        eps,
        fallback_mean=None,
        fallback_precision=None,
    ):
        num_classes, latent_dim = latent_sum.shape
        global_count = int(latent_count.sum().item())
        if global_count > 1:
            global_sum = latent_sum.sum(dim=0)
            global_outer = latent_outer.sum(dim=0)
            global_mean = global_sum / global_count
            global_cov = (global_outer - global_count * torch.outer(global_mean, global_mean)) / (global_count - 1)
            global_cov = 0.5 * (global_cov + global_cov.t())
            global_cov = global_cov + eps * torch.eye(latent_dim, device=self.device)
            default_mean = global_mean
            default_precision = self._safe_inverse(global_cov)
        else:
            default_mean = torch.zeros(latent_dim, device=self.device)

        means = default_mean.unsqueeze(0).repeat(num_classes, 1).clone()
        precisions = default_precision.unsqueeze(0).repeat(num_classes, 1, 1).clone()
        for class_id in range(num_classes):
            count = int(latent_count[class_id].item())
            if count <= 1:
                if fallback_mean is not None:
                    means[class_id] = fallback_mean[class_id]
                if fallback_precision is not None:
                    precisions[class_id] = fallback_precision[class_id]
                continue
            mean = latent_sum[class_id] / count
            cov = (latent_outer[class_id] - count * torch.outer(mean, mean)) / (count - 1)
            cov = 0.5 * (cov + cov.t())
            cov = cov + eps * torch.eye(latent_dim, device=self.device)
            means[class_id] = mean
            precisions[class_id] = self._safe_inverse(cov)
        return means.float(), precisions.float()

    def _safe_inverse(self, matrix):
        try:
            return torch.linalg.inv(matrix)
        except RuntimeError:
            return torch.linalg.pinv(matrix)

    def _masked_mean(self, values, mask):
        if not torch.any(mask):
            return values.new_tensor(0.0)
        return values[mask].mean()

    def _latent_frame_score(self, z, condition):
        machine_id = torch.argmax(condition, dim=1).long()
        source_score = self._latent_domain_score(
            z,
            machine_id,
            self.model.latent_mean_source,
            self.model.latent_precision_source,
        )
        target_score = self._latent_domain_score(
            z,
            machine_id,
            self.model.latent_mean_target,
            self.model.latent_precision_target,
        )
        return torch.minimum(source_score, target_score)

    def _latent_domain_score(self, z, machine_id, mean_bank, precision_bank):
        scores = torch.zeros(z.size(0), device=z.device)
        for class_id in torch.unique(machine_id):
            mask = machine_id == class_id
            mean = mean_bank[class_id]
            precision = precision_bank[class_id]
            delta = z[mask] - mean
            scores[mask] = torch.sum(torch.matmul(delta, precision) * delta, dim=1)
        return scores / max(z.size(1), 1)

    def _hybrid_frame_score(self, recon_score, latent_score):
        recon_z = (recon_score - self.model.score_mean[0]) / self.model.score_std[0].clamp_min(sys.float_info.epsilon)
        latent_z = (latent_score - self.model.score_mean[1]) / self.model.score_std[1].clamp_min(sys.float_info.epsilon)
        return self.args.hybrid_recon_weight * recon_z + self.args.hybrid_latent_weight * latent_z

    def _mahala_frame_score(self, data, recon_data, inv_cov_source, inv_cov_target):
        loss_source, _ = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_source,
            use_precision=True,
            reduction=False
        )
        loss_target, _ = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_target,
            use_precision=True,
            reduction=False
        )
        loss_source = loss_source.view(data.size(0), -1).mean(dim=1)
        loss_target = loss_target.view(data.size(0), -1).mean(dim=1)
        return torch.minimum(loss_source, loss_target)

    def _aggregate_frame_scores(self, frame_scores):
        if frame_scores.numel() == 0:
            return frame_scores.new_tensor(0.0)
        mean_score = frame_scores.mean()
        tail_score = torch.quantile(frame_scores, self.args.file_score_quantile)
        return (
            (1.0 - self.args.file_score_tail_weight) * mean_score
            + self.args.file_score_tail_weight * tail_score
        )

    def _aggregate_score_values(self, scores):
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            return 0.0
        mean_score = float(np.mean(scores))
        tail_score = float(np.quantile(scores, self.args.file_score_quantile))
        return (
            (1.0 - self.args.file_score_tail_weight) * mean_score
            + self.args.file_score_tail_weight * tail_score
        )

    def loss_reduction_1d(self, score):
        if score.dim() == 1:
            return score
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        if n_loss <= 0:
            return score.new_tensor(0.0)
        return torch.sum(score) / n_loss

    def train_loss_fn(self, recon_x, x):
        return F.smooth_l1_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def _save_model(self, epoch):
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': getattr(self, "loss", torch.tensor(0.0))
        }, self.checkpoint_path)

    def _score_distribution_path(self):
        if self.args.score == "MAHALA":
            return self.mahala_score_distr_file_path
        if self.args.score == "LATENT":
            return self.latent_score_distr_file_path
        if self.args.score == "HYBRID":
            return self.hybrid_score_distr_file_path
        return self.mse_score_distr_file_path

    def test(self):
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode
        csv_lines = []
        if mode:
            performance_over_all = []
            performance = []

        print("============== MODEL LOAD ==============")
        if not os.path.exists(self.model_path):
            print(f"model not found -> {self.model_path} ")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        decision_threshold = self.calc_decision_threshold(
            score_distr_file_path=self._score_distribution_path()
        )

        dir_name = "test"
        inv_cov_source, inv_cov_target = calc_inv_cov(
            model=self.model,
            device=self.device
        )
        result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"

            anomaly_score_csv = result_dir / f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            anomaly_score_list = []

            decision_result_csv = result_dir / f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            decision_result_list = []

            domain_list = [] if mode else None

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            y_pred = []
            y_true = []

            with torch.no_grad():
                y_pred, anomaly_score_list, decision_result_list, domain_list = self.eval(
                    test_loader=test_loader_tmp,
                    y_pred=y_pred,
                    anomaly_score_list=anomaly_score_list,
                    decision_result_list=decision_result_list,
                    domain_list=domain_list,
                    y_true=y_true,
                    decision_threshold=decision_threshold,
                    mode=mode,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )

            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print("anomaly score result ->  {}".format(anomaly_score_csv))

            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print("decision result ->  {}".format(decision_result_csv))

            if mode:
                y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
                y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
                y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
                y_pred_t_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]

                y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
                y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
                y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]
                y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]

                auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
                tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]).ravel()
                prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
                recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
                f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

                anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                    scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                    title=f"{section_name}_source_AUC{auc_s}"
                ))

                print("AUC (source) : {}".format(auc_s))
                print("pAUC : {}".format(p_auc))
                print("pAUC (source) : {}".format(p_auc_s))
                print("precision (source) : {}".format(prec_s))
                print("recall (source) : {}".format(recall_s))
                print("F1 score (source) : {}".format(f1_s))

                if len(y_true_t) > 0:
                    auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
                    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
                    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]).ravel()
                    prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
                    recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
                    f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["source_target"])
                    csv_lines.append([section_name.split("_", 1)[1],
                                      auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

                    performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    performance_over_all.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

                    anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_t, y_pred_t)],
                        title=f"{section_name}_target_AUC{auc_t}"
                    ))
                    print("AUC (target) : {}".format(auc_t))
                    print("pAUC (target) : {}".format(p_auc_t))
                    print("precision (target) : {}".format(prec_t))
                    print("recall (target) : {}".format(recall_t))
                    print("F1 score (target) : {}".format(f1_t))
                else:
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["single_domain"])
                    csv_lines.append([section_name.split("_", 1)[1], auc_s, p_auc, prec_s, recall_s, f1_s])

                    performance.append([auc_s, p_auc, prec_s, recall_s, f1_s])
                    performance_over_all.append([auc_s, p_auc, prec_s, recall_s, f1_s])

            print("\n============ END OF TEST FOR A SECTION ============")

        if mode:
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])

            anm_score_figdata.show_fig(
                title=self.args.model + "_" + self.args.dataset + self.model_name_suffix + self.eval_suffix + "_anm_score",
                export_dir=result_dir
            )
        else:
            return

        result_path = result_dir / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        print("results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)

    def eval(
        self,
        test_loader,
        y_pred,
        anomaly_score_list,
        decision_result_list,
        domain_list,
        y_true,
        decision_threshold,
        mode,
        inv_cov_source,
        inv_cov_target,
    ):
        for batch in test_loader:
            data, condition, _, _, _, _ = self._unpack_batch(batch)
            y_true.append(batch[1][0].item())
            basename = batch[3][0]

            recon_data, z = self.model(data, condition)
            frame_scores = self._all_frame_scores(
                data=data,
                condition=condition,
                recon_data=recon_data,
                z=z,
                inv_cov_source=inv_cov_source,
                inv_cov_target=inv_cov_target,
            )[self.args.score]
            score = float(self._aggregate_frame_scores(frame_scores).item())
            y_pred.append(score)

            anomaly_score_list.append([basename, y_pred[-1]])
            decision_result_list.append([basename, 1 if y_pred[-1] > decision_threshold else 0])

            if mode:
                domain_list.append("target" if "target" in basename else "source")
        return y_pred, anomaly_score_list, decision_result_list, domain_list


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
