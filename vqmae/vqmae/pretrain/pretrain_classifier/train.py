from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from ...base import Train
from ...data import EvaluationDataset
from torch.utils.data import Dataset
from ...model import MAE, Classifier, SpeechVQVAE, Query2Label
import matplotlib.pyplot as plt
from .follow_up_classifier import Follow
import math
torch.cuda.empty_cache()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from .asymmetric_loss import ASLSingleLabel
from einops import repeat, rearrange
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
import numpy as np
import json
import os
import gc


def compute_ser_metrics(y_true, y_pred, y_score, labels=None):
    """
    Compute required SER metrics.

    Args:
        y_true: 1D array-like of true integer/string labels
        y_pred: 1D array-like of predicted labels (same type as y_true)
        y_score: 2D array-like of shape (n_samples, n_classes) of predicted probabilities / scores.
                 Column order must match `labels`.
        labels: list of label names in the column order used in y_score. If None, inferred from y_true+y_pred sorted.

    Returns:
        metrics: dict with entries:
          - accuracy, balanced_accuracy
          - precision/recall/f1 (macro, micro, weighted)
          - per_class: dict per label with precision/recall/f1/support, specificity, AP, AUC (if computable)
          - mAP (macro-average AP), AUC_macro (macro-avg OVR), mcc
          - confusion_matrix (as pd.DataFrame)
    """


    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)
    # --- Normalize labels ---
    def normalize_label(l):
        # If numeric label, convert to name if possible
        if isinstance(l, (int, np.integer)):
            if l < len(labels):
                return labels[int(l)]
            else:
                return str(l)
        # Else assume string
        return str(l)

    y_true = [normalize_label(l) for l in y_true]
    y_pred = [normalize_label(l) for l in y_pred]

    # Convert normalized names back to indices for numeric metrics
    y_true_idx = np.array([label_to_idx[l] for l in y_true])
    y_pred_idx = np.array([label_to_idx[l] for l in y_pred])

    # Scores: ensure shape (N, n_classes)
    if y_score is None:
        # fallback: create one-hot from predictions (not ideal for AUC/AP but still produce other metrics)
        y_score = np.zeros((len(y_pred_idx), n_classes), dtype=float)
        y_score[np.arange(len(y_pred_idx)), y_pred_idx] = 1.0
    else:
        y_score = np.asarray(y_score)
        if y_score.ndim != 2 or y_score.shape[1] != n_classes:
            raise ValueError(f"y_score must be shape (N, {n_classes}); got {y_score.shape}")

    # Primary metrics
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    bal_acc = float(balanced_accuracy_score(y_true_idx, y_pred_idx))

    # Precision/Recall/F1 (per-class & averages)
    p_r_f_support = precision_recall_fscore_support(y_true_idx, y_pred_idx, labels=range(n_classes), zero_division=0)
    precision_per = p_r_f_support[0]
    recall_per = p_r_f_support[1]
    f1_per = p_r_f_support[2]
    support_per = p_r_f_support[3]

    # Averages
    precision_macro = float(np.mean(precision_per))
    recall_macro = float(np.mean(recall_per))
    f1_macro = float(np.mean(f1_per))

    p_r_f_micro = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='micro', zero_division=0)
    precision_micro, recall_micro, f1_micro = map(float, p_r_f_micro[:3])
    p_r_f_weighted = precision_recall_fscore_support(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted = map(float, p_r_f_weighted[:3])

    # Confusion matrix and specificity per class
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(n_classes))
    specificity_per = []
    per_class = {}
    for i, lab in enumerate(labels):
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(cm.sum() - (TP + FP + FN))
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity_per.append(float(specificity))

        per_class[lab] = {
            'precision': float(precision_per[i]),
            'recall': float(recall_per[i]),
            'f1': float(f1_per[i]),
            'support': int(support_per[i]),
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'specificity': float(specificity),
            'AP': None,
            'AUC': None
        }

    # MCC
    try:
        mcc = float(matthews_corrcoef(y_true_idx, y_pred_idx))
    except Exception:
        mcc = None

    # AUC (macro OVR) and per-class AUC/AP where possible
    # Binarize true labels
    y_true_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))
    # sklearn's roc_auc_score requires at least one positive label for each class to compute AUC
    aucs = []
    aps = []
    for i in range(n_classes):
        y_true_i = y_true_bin[:, i]
        y_score_i = y_score[:, i]
        # AUC
        try:
            auc_i = roc_auc_score(y_true_i, y_score_i)
        except Exception:
            auc_i = None
        # Average Precision
        try:
            ap_i = average_precision_score(y_true_i, y_score_i)
        except Exception:
            ap_i = None
        per_class[labels[i]]['AUC'] = float(auc_i) if auc_i is not None else None
        per_class[labels[i]]['AP'] = float(ap_i) if ap_i is not None else None
        if auc_i is not None:
            aucs.append(auc_i)
        if ap_i is not None:
            aps.append(ap_i)

    auc_macro = float(np.mean(aucs)) if len(aucs) > 0 else None
    mAP_macro = float(np.mean(aps)) if len(aps) > 0 else None

    # Build metric dict
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'auc_macro': auc_macro,
        'mAP_macro': mAP_macro,
        'per_class': per_class,
        'specificity_per_class': dict(zip(labels, specificity_per)),
        'confusion_matrix': {
            'labels': labels,
            'matrix': cm.tolist()
        }
    }
    return metrics

# Checkpointing helper (pseudo code-style saver to integrate into training loop)
def save_checkpoint_if_best(model, optimizer, epoch, metrics, checkpoint_dir, primary_metric_name='balanced_accuracy', maximize=True):
    """
    Save model checkpoint when primary metric improves.

    - metrics: dict returned from compute_ser_metrics on validation set.
    - checkpoint_dir: directory to save checkpoints
    - primary_metric_name: string key in metrics to use for selecting the best checkpoint (default 'balanced_accuracy')
    - maximize: True if higher is better (True for accuracy/balanced_accuracy), False if lower is better.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_meta_path = os.path.join(checkpoint_dir, 'best_checkpoint_meta.json')
    # load existing best
    if os.path.exists(best_meta_path):
        with open(best_meta_path, 'r') as f:
            best_meta = json.load(f)
    else:
        best_meta = {'best_value': None, 'best_epoch': None, 'best_file': None}

    cur_value = metrics.get(primary_metric_name, None)
    is_better = False
    if cur_value is None:
        is_better = False
    else:
        if best_meta['best_value'] is None:
            is_better = True
        else:
            if maximize:
                is_better = cur_value > best_meta['best_value']
            else:
                is_better = cur_value < best_meta['best_value']

    if is_better:
        # save model - adapt this to your framework (torch.save, tf.keras.Model.save, etc.)
        filename = f"checkpoint_epoch{epoch}_{primary_metric_name}_{cur_value:.6f}.pt"
        filepath = os.path.join(checkpoint_dir, filename)
        # Example for PyTorch:
        try:
            import torch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                'metrics': metrics
            }, filepath)
        except Exception:
            # fallback: if not PyTorch, try model.save() for TF or other saving logic
            try:
                model.save(filepath)
            except Exception as e:
                # last resort: save only metrics
                with open(filepath + '.metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

        # update best meta
        best_meta['best_value'] = float(cur_value)
        best_meta['best_epoch'] = int(epoch)
        best_meta['best_file'] = filename
        with open(best_meta_path, 'w') as f:
            json.dump(best_meta, f, indent=2)

        # also save human-readable metrics snapshot
        metrics_snapshot_path = os.path.join(checkpoint_dir, f"metrics_epoch{epoch}.json")
        with open(metrics_snapshot_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    return is_better


class Classifier_Train(Train):
    def __init__(self, mae: MAE,
                 vqvae: SpeechVQVAE,
                 training_data: Dataset,
                 test_data: Dataset,
                 config_training: dict = None, follow: bool = True,
                 query2emo: bool = False):
        super().__init__()
        if "device" in config_training and config_training["device"] in ["cuda", "cuda:0", "cpu"]:
            self.device = torch.device(config_training["device"] if torch.cuda.is_available() or config_training["device"] == "cpu" else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.device = torch.device(config_training['device'])
        """ Model """
        if query2emo:
            self.model = Query2Label(encoder=mae.encoder, num_classes=8)
        else:
            self.model = Classifier(encoder=mae.encoder, num_classes=8)
        self.model.to(self.device)
        self.vqvae = vqvae
        self.vqvae.to(self.device)

        """ Dataloader """
        if config_training is None:
            config_training = {}
        batch_size = int(config_training.get("batch_size", 16))
        num_workers = int(config_training.get("num_workers", 0))
        pin_memory = bool(config_training.get("pin_memory", True))
        
        self.training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        self.validation_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config_training['lr']/10.0,
                                           betas=(0.9, 0.95),
                                           weight_decay=config_training["weight_decay"])
        lr_func = lambda epoch: min((epoch + 1) / (40 + 1e-8),
                                    0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)

        """ Loss """
        # weights = training_data.get_weights(num_class=8)
        # self.criterion = torch.nn.CrossEntropyLoss(reduction="mean", weight=weights.to(self.device))
        self.criterion = ASLSingleLabel(reduction="mean")
        self.acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
        self.best_acc = 0.0
        self.best_f1 = 0.0

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()

        """ Follow """
        if follow:
            self.follow = Follow("classifier", dir_save=r"/net/tscratch/people/plgmarbar/ravdess/vsqmae_checkpoints", variable=vars(self.model))

    @staticmethod
    def to_tube(input, size_patch=4, depth_t=5):
        c1 = int(input.shape[-1] / size_patch)
        t1 = input.shape[1] // depth_t
        input = rearrange(input, 'b (t1 t2) (c1 l1) -> b (t1 c1) (l1 t2)', t1=t1, t2=depth_t, c1=c1, l1=size_patch)
        return input


    def one_epoch(self):
        self.model.train()
        losses = []
        acces = []
        for input, label in tqdm(iter(self.training_loader)):
            self.optimizer.zero_grad()
            
            self.step_count += 1
            input = input.to(self.device)
            label = label.to(self.device)
            # input = self.to_tube(input, depth_t=10, size_patch=4)
            logits = self.model(input)
            loss = self.criterion(logits, label)
            acc = self.acc_fn(logits, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            acces.append(acc.item())
        return losses, acces

    # def fit(self):
    #     for e in range(self.config_training["total_epoch"]):
    #         losses, acces = self.one_epoch()
    #         losses_test, acces_test, f1_test = self.eval()
    #         self.lr_scheduler.step()
    #         avg_loss_train = sum(losses) / len(losses)
    #         avg_train_acc = sum(acces) / len(acces)
    #         avg_loss_val = sum(losses_test) / len(losses_test)
    #         avg_test_acc = sum(acces_test) / len(acces_test)
    #         self.parameters = dict(model=self.model.state_dict(),
    #                                optimizer=self.optimizer.state_dict(),
    #                                scheduler=self.lr_scheduler.state_dict(),
    #                                epoch=e,
    #                                loss=avg_loss_train)
    #         print(
    #             f'In epoch {e}, average traning loss is {avg_loss_train:.3f}.'
    #             f' and average validation loss is {avg_loss_val:.3f}')
    #         print(
    #             f'\t - average accuracy is {avg_train_acc:.3f}.'
    #             f' and average validation accuracy is {avg_test_acc:.3f} and F1 score is  {f1_test:.3f}')
    #         self.follow(epoch=e,
    #                     loss_train=avg_train_acc,
    #                     loss_validation=avg_test_acc,
    #                     parameters=self.parameters,
    #                     f1_loss=f1_test)

    #         if avg_test_acc > self.best_acc or f1_test > self.best_f1:
    #             # update both bests when improved
    #             if avg_test_acc > self.best_acc:
    #                 self.best_acc = avg_test_acc
    #             if f1_test > self.best_f1:
    #                 self.best_f1 = f1_test
    #             best_epoch = e

    #     # return (best accuracy, best f1) so external caller can append/aggregate
    #     return self.best_acc, self.best_f1
        # return self.follow.best_loss, self.follow.best_f1

    # def eval(self):
    #     self.model.eval()
    #     losses = []
    #     acces = []
    #     y_true = torch.empty((0,), dtype=torch.long)
    #     y_pred = torch.empty((0,), dtype=torch.long)
    #     with torch.no_grad():
    #         for input, label in tqdm(iter(self.validation_loader)):
    #             # y_true = torch.cat((y_true, label), dim=0)
    #             if isinstance(label, torch.Tensor):
    #                 y_true = torch.cat((y_true, label.to(torch.long).cpu()), dim=0)
    #             else:
    #                 # safety fallback
    #                 y_true = torch.cat((y_true, torch.tensor(label, dtype=torch.long)), dim=0)

    #             input = input.to(self.device)
    #             label = label.to(self.device)
    #             # input = self.to_tube(input, depth_t=10, size_patch=4)
    #             logits = self.model(input)
    #             # y_pred = torch.cat((y_pred, logits.argmax(dim=-1)), dim=0)
    #             preds = logits.argmax(dim=-1)  # on device

    #             # accumulate predictions on CPU
    #             y_pred = torch.cat((y_pred, preds.cpu().to(torch.long)), dim=0)

    #             loss = self.criterion(logits, label)
    #             acc = self.acc_fn(logits, label)
    #             losses.append(loss.item())
    #             acces.append(acc.item())
    #     # labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    #     f1 = f1_score(y_true.numpy(), y_pred.cpu().detach().numpy(), average="weighted")
    #     # if sum(acces) / len(acces) > self.best_acc:
    #     #     self.best_acc = sum(acces) / len(acces)
    #     #     labels = ["W", "L", "E", "A", "F", "T", "N"]
    #     #     # cm = confusion_matrix(y_true.numpy(), y_pred.cpu().detach().numpy())
    #     #     plt.figure(figsize=(15, 15))
    #     #     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #     #     cm = confusion_matrix(y_true.numpy(), y_pred.cpu().detach().numpy())
    #     #     num_classes = cm.shape[0]
    #     #     labels = [f"Class {i}" for i in range(num_classes)]
    #     #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #     #     disp.plot()
    #     #     plt.savefig(f'{self.follow.path}/matrix_confusion.png')
    #     #     plt.savefig(f'{self.follow.path}/matrix_confusion.svg')
    #     if sum(acces) / len(acces) > self.best_acc:
    #         self.best_acc = sum(acces) / len(acces)
        
    #         # Force confusion matrix to always have 8 classes
    #         # cm = confusion_matrix(
    #         #     y_true.numpy(),
    #         #     y_pred.cpu().detach().numpy(),
    #         #     labels=list(range(8))
    #         # )
        
    #         # labels = [f"Class {i}" for i in range(8)]
    #         # Build confusion matrix
    #         cm = confusion_matrix(y_true.numpy(), y_pred.cpu().numpy())
    #         num_classes = cm.shape[0]
    
    #         # Dynamically generate labels based on class count
    #         labels = [f"Class {i}" for i in range(num_classes)]
    #         plt.figure(figsize=(15, 15))
    #         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #         disp.plot()
    #         plt.savefig(f'{self.follow.path}/matrix_confusion.png')
    #         plt.savefig(f'{self.follow.path}/matrix_confusion.svg')
    #     return losses, acces, f1

    def load(self, path: str = "", optimizer: bool = True):
         print("LOAD [", end="")
         # checkpoint = torch.load(path)
         checkpoint = torch.load(path, map_location=self.device)
         self.model.load_state_dict(checkpoint['model'])
         if optimizer:
             self.optimizer.load_state_dict(checkpoint['optimizer'])
             self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
         self.load_epoch = checkpoint['epoch']
         loss = checkpoint['loss']
         print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")

    # def plot_3D(self):
    #     pca = TSNE(n_components=3)
    #     features = torch.tensor([]).to(self.device)
    #     labels = torch.tensor([])
    #     with torch.no_grad():
    #         for img, label in tqdm(iter(self.training_loader)):
    #             labels = torch.cat((labels, label), dim=0)
    #             img = img.to(self.device)
    #             # indices = self.vqvae.get_codebook_indices(img)  # .cpu().detach().numpy()
    #             # indices = rearrange(indices, 'b (h w) -> b h w', h=64, w=64)
    #             # indices = rearrange(indices, 'b (h c1) (w c2) -> b (h w) (c1 c2)', c1=4, c2=4)
    #             cls = self.model.get_cls(img)
    #             features = torch.cat((features, cls), dim=0)
    #     features = features.cpu().detach().numpy()
    #     labels = labels.numpy()
    #     features = pca.fit_transform(features)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     # name_labels = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]  # Jaffed dataset
    #     name_labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    #     colors = ['red', 'cyan', 'violet', 'pink', 'olive', 'sienna', 'navy']
    #     for i in range(7):
    #         indx = labels == i
    #         data = features[indx]
    #         ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i], label=name_labels[i])
    #     plt.title('Matplot 3d scatter plot')
    #     plt.legend(loc=2)
    #     plt.show()


    # def eval(self):
    #     self.model.eval()
    #     losses, acces = [], []
    #     y_true = []
    #     y_pred = []
    #     y_score = []

    #     with torch.no_grad():
    #         for input, label in tqdm(iter(self.validation_loader)):
    #             input = input.to(self.device)
    #             label = label.to(self.device)

    #             logits = self.model(input)
    #             probs = torch.softmax(logits, dim=-1)

    #             loss = self.criterion(logits, label)
    #             acc = self.acc_fn(logits, label)

    #             losses.append(loss.item())
    #             acces.append(acc.item())

    #             y_true.extend(label.cpu().numpy().tolist())
    #             y_pred.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
    #             y_score.extend(probs.cpu().numpy())

    #     # === Compute full SER metrics ===
    #     labels = [f"Class {i}" for i in range(8)]  # Adjust to match your emotion labels
    #     metrics = compute_ser_metrics(
    #         y_true=np.array(y_true),
    #         y_pred=np.array(y_pred),
    #         y_score=np.array(y_score),
    #         labels=labels
    #     )

    #     # === Save confusion matrix figure ===
    #     cm = np.array(metrics['confusion_matrix']['matrix'])
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #     plt.figure(figsize=(12, 12))
    #     disp.plot(cmap='Blues')
    #     plt.title("Confusion Matrix (Validation)")
    #     plt.savefig(f'{self.follow.path}/matrix_confusion.png')
    #     plt.close()

    #     # === Save metrics to JSON for tracking ===
    #     metrics_path = os.path.join(self.follow.path, 'validation_metrics.json')
    #     with open(metrics_path, 'w') as f:
    #         json.dump(metrics, f, indent=2)

    #     # === Track best model (optional) ===
    #     save_checkpoint_if_best(
    #         model=self.model,
    #         optimizer=self.optimizer,
    #         epoch=self.load_epoch,
    #         metrics=metrics,
    #         checkpoint_dir=self.follow.path,
    #         primary_metric_name='balanced_accuracy',
    #         maximize=True
    #     )

    #     avg_f1 = metrics['f1_weighted']
    #     return losses, acces, avg_f1
    def fit(self):
        detailed_eval_interval = 5  # compute full metrics every 5 epochs
        all_epoch_logs = []

        for e in range(self.config_training["total_epoch"]):
            # --- Training phase ---
            losses, acces = self.one_epoch()
            avg_loss_train = np.mean(losses)
            avg_train_acc = np.mean(acces)

            # --- Validation phase ---
            # Always compute loss + balanced accuracy, but not full metrics every epoch
            losses_val, acces_val, bal_acc_val = self.eval(e, detailed=(e % detailed_eval_interval == 0))

            avg_loss_val = np.mean(losses_val)
            avg_val_acc = np.mean(acces_val)

            # Update LR
            self.lr_scheduler.step()

            print(
                f"[Epoch {e:03d}] Train Loss: {avg_loss_train:.3f}, "
                f"Train Acc: {avg_train_acc:.3f}, "
                f"Val Loss: {avg_loss_val:.3f}, "
                f"Val Acc: {avg_val_acc:.3f}, "
                f"Val BalAcc: {bal_acc_val:.3f}"
            )

            # Track best accuracy / balanced accuracy
            if bal_acc_val > self.best_acc:
                self.best_acc = bal_acc_val
                self.best_f1 = self.best_f1  # unchanged unless updated in eval()

            # Lightweight per-epoch summary (small JSONL)
            epoch_summary = {
                "epoch": e,
                "train_loss": avg_loss_train,
                "train_accuracy": avg_train_acc,
                "val_loss": avg_loss_val,
                "val_accuracy": avg_val_acc,
                "val_balanced_accuracy": bal_acc_val,
            }
            all_epoch_logs.append(epoch_summary)

            # Append small log to file incrementally
            log_path = os.path.join(self.follow.path, "training_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_summary) + "\n")

            gc.collect()

        print(f"Best balanced accuracy: {self.best_acc:.4f}")
        return self.best_acc, self.best_f1


    def eval(self, epoch, detailed=False):
        self.model.eval()
        losses, acces = [], []
        y_true, y_pred, y_score = [], [], []

        with torch.no_grad():
            for input, label in tqdm(iter(self.validation_loader), desc=f"Validation Epoch {epoch}"):
                input = input.to(self.device)
                label = label.to(self.device)

                logits = self.model(input)
                probs = torch.softmax(logits, dim=-1)

                loss = self.criterion(logits, label)
                acc = self.acc_fn(logits, label)

                losses.append(loss.item())
                acces.append(acc.item())

                y_true.extend(label.cpu().numpy().tolist())
                y_pred.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
                y_score.extend(probs.cpu().numpy())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        # Always compute balanced accuracy
        bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)

        # --- Only compute detailed metrics periodically ---
        if detailed:
            labels = [f"Class {i}" for i in range(8)]
            metrics = compute_ser_metrics(
                y_true=y_true_np,
                y_pred=y_pred_np,
                y_score=np.array(y_score),
                labels=labels
            )

            # --- Confusion matrix summary (normalized) ---
            cm = np.array(metrics['confusion_matrix']['matrix'])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_summary = {labels[i]: {labels[j]: float(cm_norm[i, j]) for j in range(len(labels))} for i in range(len(labels))}
            metrics['confusion_matrix_summary'] = cm_summary
            del metrics['confusion_matrix']  # remove full matrix to save space

            # --- Append to one JSONL file (incremental) ---
            detailed_metrics_path = os.path.join(self.follow.path, "detailed_metrics.jsonl")
            with open(detailed_metrics_path, "a") as f:
                record = {"epoch": epoch, **metrics}
                f.write(json.dumps(record) + "\n")

            # --- Save checkpoint only on detailed evals ---
            save_checkpoint_if_best(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_dir=self.follow.path,
                primary_metric_name='balanced_accuracy',
                maximize=True
            )

            # Update best f1 if improved
            if metrics["f1_weighted"] > self.best_f1:
                self.best_f1 = metrics["f1_weighted"]

            print(f"Detailed metrics computed and saved for epoch {epoch}")

            # Clean memory
            del metrics, cm, cm_norm, cm_summary
            gc.collect()

        return losses, acces, bal_acc


