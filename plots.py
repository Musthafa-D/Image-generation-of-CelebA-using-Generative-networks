import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import torch
import math
from sklearn.manifold import TSNE
from ccbdl.evaluation.plotting.base import GenericPlot
from captum.attr import visualization as viz
from captum.attr import Saliency, GuidedBackprop, InputXGradient, Deconvolution, Occlusion
from ccbdl.utils.logging import get_logger
from ccbdl.evaluation.plotting import graphs
from sklearn.metrics import confusion_matrix
from networks import CNN
from torch.utils.data import DataLoader, Dataset
from ccbdl.config_loader.loaders import ConfigurationLoader
from matplotlib.colors import LinearSegmentedColormap
from ccbdl.utils import DEVICE
import matplotlib.ticker as ticker
from setsize import set_size

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


class TimePlot(GenericPlot):
    def __init__(self, learner):
        super(TimePlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating time plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        xs, ys = zip(*self.learner.data_storage.get_item("Time", batch=True))
        
        if self.learner.learner_config["model"] == 'CGAN':
            ax.plot(xs, [y - ys[0]for y in ys], label="cgan_train_time")
        else:
            ax.plot(xs, [y - ys[0]for y in ys], label="gan_train_time")
        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$t$', fontsize=14)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(ys))
        ax.legend()
        
        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "time_plot"))
        return figs, names


class Loss_plot(GenericPlot):
    def __init__(self, learner):
        super(Loss_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating loss plot")

    def consistency_check(self):
        return True
    
    def graph_multix_multiy(self, xs, ys, colors, alphas, *args, **kwargs):
        """
        Function to plot multiple y-values over different x-values.
    
        Args:
            xs (iterable of iterables): Each element of this 2-d iterable contains the full list of x-values.
            ys (iterable of iterables): Each element of this 2-d iterable contains the full list of y-values.
            colors (list): List of colors for each plot.
            alphas (list): List of alpha (transparency) values for each plot.
            *args (iterable): Default support for args (not used).
            **kwargs (dict): Default support for kwargs (not used).
    
        Returns:
            fig (matplotlib.figure.Figure): The generated figure.
        """
        fig = plt.figure()
        
        for idx in range(len(ys)):
            if "labels" in kwargs:
                label = kwargs["labels"][idx]
            else:
                label = str(idx)
                
            x = xs[idx]
            y = ys[idx]
            plt.plot(x, y, label=label, color=colors[idx], alpha=alphas[idx])
        
        plt.xlabel("$B$", fontsize=14)
        plt.ylabel('$\\mathcal{L}$', fontsize=14)
        
        plt.xlim(left=0, right=max(x))
        plt.ylim(bottom=0)
    
        plt.legend()
        plt.grid()
        plt.tight_layout()
    
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        
        return fig

    def plot(self):
        figs = []
        names = []
        
        xs = []
        ys = []
        
        plot_names = []
        
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("gen_loss")
        xs.append(x)
        ys.append(y)
        plot_names.append("$\\mathcal{L}_{\\mathrm{gen}}$")
        xs.append(x[: -49])
        ys.append(graphs.moving_average(y, 50))
        plot_names.append("$\\mathcal{L}_{\\mathrm{gen\\_avg}}$")
        
        
        x = self.learner.data_storage.get_item("batch") 
        y = self.learner.data_storage.get_item("dis_loss")
        xs.append(x)
        ys.append(y)
        plot_names.append("$\\mathcal{L}_{\\mathrm{dis}}$")
        xs.append(x[: -49])
        ys.append(graphs.moving_average(y, 50))
        plot_names.append("$\\mathcal{L}_{\\mathrm{dis\\_avg}}$")
        
        figs.append(self.graph_multix_multiy(xs = xs,
                                           ys = ys,
                                           labels = plot_names,
                                           xlim = (0, min([x[-1] for x in xs])),
                                           # ylim = (0, max([max(y) for y in ys])),
                                           ylim = (0, 4),
                                           colors = ['red', 'darkred', 'blue', 'darkblue'],
                                           alphas = [0.5, 0.5, 0.5, 0.5]
                                           ))
        names.append(os.path.join("plots", "losses"))
        return figs, names


class Psnr_plot(GenericPlot):
    def __init__(self, learner):
        super(Psnr_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating psnr score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("psnr_score")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the PSNR scores
        ax.plot(x, y)
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{psnr}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "psnr_scores")]
        return figs, names
    

class Ssim_plot(GenericPlot):
    def __init__(self, learner):
        super(Ssim_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating psnr score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("ssim_score")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the PSNR scores
        ax.plot(x, y)
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{ssim}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "ssim_scores")]
        return figs, names


class Fid_plot_nlrl(GenericPlot):
    def __init__(self, learner):
        super(Fid_plot_nlrl, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating fid score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("fid_score_nlrl")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the FID scores
        ax.plot(x, y, label="CNN_RGB")
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{fid}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "fid_scores_nlrl")]
        return figs, names
    

class Fid_plot_linear(GenericPlot):
    def __init__(self, learner):
        super(Fid_plot_linear, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating fid score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("fid_score_linear")

        # Create a single plot (no subplots)
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        # Plot the FID scores
        ax.plot(x, y, label="CNN_RGB")
        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{fid}$", fontsize=14)
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(y))
        ax.grid(True)
        ax.legend()

        # Finalize the layout
        fig.tight_layout()

        # Prepare the output
        figs = [fig]
        plt.close(fig)
        names = [os.path.join("plots", "fid_scores_linear")]
        return figs, names
        
        
class Image_generation(GenericPlot):
    def __init__(self, learner):
        super(Image_generation, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator for certain epochs")
        
        self.style = {}

    def consistency_check(self):
        return True
    
    def grid_2d(self, imgs, preds, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Args:
            original (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            reconstructed (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns:
            None.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):        
            ax = plt.subplot(rows, cols, i+1)
            # get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)

            ax.set_title(f"{i}, Prediction: {preds[i]:.3f}")
            # remove axes
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            
            # show img
            plt.imshow(img)
        plt.tight_layout()
        return fig
    
    def grid_2d_labels(self, imgs, preds, labels, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Args:
            original (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            reconstructed (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].
            figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).
            labels (torch.Tensor): Tensor with N number of int values.

        Returns:
            None.

        """
        
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):        
            ax = plt.subplot(rows, cols, i+1)
            # get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)

            ax.set_title(f"{i}, Labels: {labels[i]}\nPrediction: {preds[i]:.3f}")
            # remove axes
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            
            # show img
            plt.imshow(img)
            
        plt.tight_layout()
        return fig

    def plot(self):
        generated_images = self.learner.data_storage.get_item("generated_images")
        predictions = self.learner.data_storage.get_item("predictions_gen")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        
        if self.learner.learner_config["model"] == 'CGAN':
            labels = self.learner.data_storage.get_item("test_labels")
            
        total = len(epochs)

        figs = []
        names = []

        for idx in range(total):
            generated_images_per_epoch = generated_images[idx]
            num = generated_images_per_epoch.size(0)
            
            self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (20, 20)})
            
            epoch = epochs[idx]
            prediction_per_epoch= predictions[idx]
            
            if self.learner.learner_config["model"] == 'CGAN':
                label = labels[idx].cpu().detach().numpy()
                figs.append(self.grid_2d_labels(imgs=generated_images_per_epoch, 
                                                  labels=label, 
                                                  preds=prediction_per_epoch, 
                                                  **self.style))
            else:
                figs.append(self.grid_2d(imgs=generated_images_per_epoch, 
                                           preds=prediction_per_epoch,
                                           **self.style))

            names.append(os.path.join("plots", "generated_images", f"epoch_{epoch}"))
        return figs, names


class Confusion_matrix_gan(GenericPlot):
    def __init__(self, learner):
        super(Confusion_matrix_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating confusion matrix for real vs. fake predictions")

    def consistency_check(self):
        return True

    def plot(self):
        names = []
        figs = []
        
        fake_probs = self.learner.data_storage.get_item("predictions_fake")
        real_probs = self.learner.data_storage.get_item("predictions_real")
        
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        threshold = self.learner.threshold
        
        total = len(epochs)
        
        for idx in range(total):
            fig, ax = plt.subplots(figsize=(10, 6))
            epoch = epochs[idx]
            
            # Extract and flatten the predictions for the current epoch
            fake_probs_per_epoch = fake_probs[idx].flatten()
            real_probs_per_epoch = real_probs[idx].flatten()

            # Convert probabilities to binary predictions using a threshold eg 0.5
            fake_predictions = (fake_probs_per_epoch < threshold).astype(int)
            real_predictions = (real_probs_per_epoch > threshold).astype(int)

            # Concatenate predictions and true labels
            predictions = np.concatenate([fake_predictions, real_predictions])
            correct_labels = np.concatenate([np.zeros_like(fake_predictions), np.ones_like(real_predictions)])
            
            # Compute confusion matrix
            matrix = confusion_matrix(correct_labels, predictions)

            # Plot the confusion matrix using seaborn's heatmap
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', ax=ax,
                        xticklabels=['Predicted Fake', 'Predicted Real'],
                        yticklabels=['Actual Fake', 'Actual Real'])
            
            figs.append(fig)
            names.append(os.path.join("plots", "confusion_matrices", "gan_training_based", f"epoch_{epoch}"))
        return figs, names


class Tsne_plot_classifier(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on classifier's features and classifier's decisions")
    
    def consistency_check(self):
        return True
    
    def load_classifier(self):
        config = ConfigurationLoader().read_config("config.yaml")
        classifier_config = config["classifier"]
        
        classifier = CNN("Classifier to classify both real and fake images", 
                      **classifier_config).to(self.learner.device)
        
        checkpoint = torch.load('cnn_net_best.pt')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        return classifier
    
    def get_features(self, classifier, imgs):
        activation = {}
        
        def get_activation(name):
            def hook(classifier, inp, output):
                activation[name] = output.detach()
            return hook
        
        # Register the hook
        handle = classifier.model[-4].register_forward_hook(get_activation('conv'))
        _ = classifier(imgs)
        
        # Remove the hook
        handle.remove()
        return activation['conv']
    
    def compute_tsne(self, features):
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features)
        return tsne_results
    
    def process_images(self, data_loader, classifier, cat):
        all_features = []
        all_labels = []
        
        for imgs in data_loader:  # data_loader now provides both images and labels
            outputs = classifier(imgs)
            sigmoid = torch.nn.Sigmoid()
            predicted_probs = sigmoid(outputs)  # Use sigmoid for multi-label classification
            predicted_labels = (predicted_probs > 0.5).int()  # Convert probabilities to binary labels
            features = self.get_features(classifier, imgs)
            features = features.view(features.size(0), -1)  # Flatten the features
            all_features.append(features)
            all_labels.append(predicted_labels)
            
        # Concatenate all the features and labels from the batches
        if cat == 1:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    def plot(self):
        # Load the classifier
        classifier = self.load_classifier()
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        # Setting concatenation true by initializing value as 1
        cat = 1
        config = ConfigurationLoader().read_config("config.yaml")
        data_config = config["data"]
    
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        
        real_images = total_real_images[-1]
        real_dataset = ImageTensorDataset(real_images)
        real_data_loader = DataLoader(real_dataset, batch_size=data_config["batch_size"], shuffle=False)
        real_features, real_labels = self.process_images(real_data_loader, classifier, cat)
    
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        
        fake_images = total_fake_images[-1]
        fake_dataset = ImageTensorDataset(fake_images)
        fake_data_loader = DataLoader(fake_dataset, batch_size=data_config["batch_size"], shuffle=False)
        fake_features, fake_labels = self.process_images(fake_data_loader, classifier, cat)
    
        # Combine features for t-SNE
        combined_features = torch.cat([real_features, fake_features], dim=0)
        tsne_results = self.compute_tsne(combined_features.cpu().numpy())
    
        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:len(real_features)]
        fake_tsne = tsne_results[len(real_features):]
    
        # Plotting
        figs, names = [], []
        for label_index, label_name in enumerate(label_names):
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = real_labels[:, label_index].cpu().numpy()
            real_count = real_indices.sum()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices == 1, 0],  # Plot images having the attribute
                y=real_tsne[real_indices == 1, 1], 
                label=f"Real {label_name}, (Count: {real_count})", 
                alpha=0.5
            )
            # Fake images scatter plot
            fake_indices = fake_labels[:, label_index].cpu().numpy()
            fake_count = fake_indices.sum()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices == 1, 0],  # Plot images having the attribute
                y=fake_tsne[fake_indices == 1, 1], 
                label=f"Fake {label_name}, (Count: {fake_count})", 
                alpha=0.5
            )
            ax.set_title(f"t-SNE visualization for attribute '{label_name}'")
            ax.legend()
            figs.append(fig)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_classifier", f"attribute_{label_name}"))   
        return figs, names


class Tsne_plot_dis_gan(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_dis_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on gan's discriminator's features and discriminator's decisions")
    
    def consistency_check(self):
        return True
    
    def get_features(self, layer_num, real_images, fake_images, discriminator):
        features = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator[layer_num].register_forward_hook(hook)

        # Process real images through the discriminator
        discriminator(real_images)

        # Process fake images through the discriminator
        discriminator(fake_images)

        handle.remove()  # Remove the hook
        return torch.cat(features)
    
    def compute_tsne(self, features):
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        real_images = total_real_images[-1]
        
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-1]
        
        self.learner._load()  # Load the best epoch's model's discriminator with the respective weights
        discriminator = self.learner.model.discriminator
        layer_num = -4
        
        # Extract features from the discriminator
        features = self.get_features(layer_num, real_images, fake_images, discriminator)
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        # Compute t-SNE
        tsne_results = self.compute_tsne(features)
        
        half = len(tsne_results) // 2

        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:half]
        fake_tsne = tsne_results[half:]
        
        # Plotting
        figs, names = [], []
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Scatter plot for real images
        ax.scatter(real_tsne[:, 0], real_tsne[:, 1], label="Real Images", alpha=0.5, color='blue')
        
        # Scatter plot for fake images
        ax.scatter(fake_tsne[:, 0], fake_tsne[:, 1], label="Fake Images", alpha=0.5, color='red')
        
        ax.set_title("t-SNE visualization of GAN's Discriminator Features")
        ax.legend()
        
        figs.append(fig)
        names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_gan_discriminator", "combined_plot"))   
        return figs, names

    
class Tsne_plot_dis_cgan(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_dis_cgan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on cgan's discriminator's features and discriminator's decision")
    
    def consistency_check(self):
        return True
    
    def get_features(self, layer_num, real_images, fake_images, discriminator, labels):
        features = []
    
        def hook(discriminator, inp, output):
            features.append(output.detach())
    
        # Attach the hook to the desired layer
        handle = discriminator.dis[layer_num].register_forward_hook(hook)
    
        # Process the whole batch of real and fake images through the discriminator
        discriminator(real_images, labels)
        discriminator(fake_images, labels)
    
        handle.remove()  # Remove the hook
        return torch.cat(features)
    
    def compute_tsne(self, features):
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        real_images = total_real_images[-1]
        
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-1]
        
        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-1]
        
        self.learner._load()  # Load the best epoch's model's discriminator with the respective weights
        discriminator = self.learner.model.discriminator
        layer_num = 0
        
        features = self.get_features(layer_num, real_images, fake_images, discriminator, labels)
        features = features.view(features.size(0), -1)  # Flatten the features
        
        # Compute t-SNE
        tsne_results = self.compute_tsne(features)
        
        half = len(tsne_results) // 2

        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:half]
        fake_tsne = tsne_results[half:]
        
        # Attributes list for CelebA
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        # Plotting
        figs, names = [], []
        for label_index, label_name in enumerate(label_names):
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = (labels[:, label_index] == 1).cpu().numpy()
            real_count = real_indices.sum()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"Real {label_name}, (Count: {real_count})", 
                color="blue",
                alpha=0.5
            )
            # Fake images scatter plot
            fake_indices = (labels[:, label_index] == 1).cpu().numpy()
            fake_count = fake_indices.sum()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"Fake {label_name}, (Count: {fake_count})", 
                color="red",
                alpha=0.5
            )
            ax.set_title(f"t-SNE visualization for attribute '{label_name}'")
            ax.legend()
            figs.append(fig)
            names.append(os.path.join("plots", "analysis_plots", "tsne_plots", "feature_based_cgan_discriminator", f"attribute_{label_name}"))   
        return figs, names


class Attribution_plots(GenericPlot):
    def __init__(self, learner):
        super(Attribution_plots, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for some images of celeba")

    def consistency_check(self):
        return True

    def safe_visualize(self, attr, original_image, title, fig, ax, label, img_name, types, cmap):
        if not (attr == 0).all():
            viz.visualize_image_attr(attr, 
                                     original_image=original_image, 
                                     method='heat_map', 
                                     sign='all', 
                                     show_colorbar=True, 
                                     title=title, 
                                     plt_fig_axis=(fig, ax),
                                     cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        names = []
        figs = []
        max_images_per_plot = 4  # Define a constant for the maximum number of images per plot

        self.learner._load()  
        model = self.learner.model.discriminator # Load the best epoch's model's discriminator with the respective weights
        
        # Custom cmap for better visulaization
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])

        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        fake_preds = self.learner.data_storage.get_item("predictions_fake")
        real_preds = self.learner.data_storage.get_item("predictions_real")
        
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        if self.learner.learner_config["model"] == 'CGAN':
            labels_list = self.learner.data_storage.get_item("labels")

        for types in ["real", "fake"]:
            inputs = real_images[-1][:40] if types == 'real' else fake_images[-1][:40]
            preds = real_preds[-1][:40] if types == 'real' else fake_preds[-1][:40]
    
            inputs.requires_grad = True  # Requires gradients set true
            
            if self.learner.learner_config["model"] == 'CGAN':
                labels = labels_list[-1][:40]
                # Get attribution maps for different techniques
                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps(model, inputs, labels)
            
            else:
                # Get attribution maps for different techniques
                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps(model, inputs)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]

            # Process all input images
            total_indices = list(range(inputs.shape[0])) # to avoid memory related errors
            chunks = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for chunk in chunks:
                num_rows = len(chunk)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 7 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in chunk:
                    img = (inputs[idx].cpu().detach().permute(1, 2, 0)).numpy()
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    pred = preds[idx]
                    
                    if self.learner.learner_config["model"] == 'CGAN':
                        label = labels[idx].cpu().detach().numpy()
                        # Convert binary label and prediction to named labels
                        actual_labels = [name for i, name in enumerate(label_names) if label[i] == 1]
                        actual_labels_str = ",\n ".join(actual_labels)
            
                    # Retrieve the attribution maps for the current image
                    results = [
                                np.transpose(saliency_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                np.transpose(guided_backprop_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                np.transpose(input_x_gradient_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                np.transpose(deconv_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                np.transpose(occlusion_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                ]

                    # Display the original image
                    axs[count, 0].imshow(img)
                    axs[count, 0].set_title(f"Prediction: {pred:.3f}")
                    if self.learner.learner_config["model"] == 'CGAN':
                        axs[count, 0].set_title(f"Labels: {label}\nActive Labels: {actual_labels_str}\nPrediction: {pred:.3f}", fontsize=10)
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        
                        if self.learner.learner_config["model"] == 'CGAN':
                            self.safe_visualize(res, img, title, fig, axs[count, col + 1], label, None, types, cmap)
                        
                        else:
                            # Call the visualization function, passing None for label and img_name since they are not applicable
                            self.safe_visualize(res, img, title, fig, axs[count, col + 1], None, None, types, cmap)
            
                    count += 1
            
                # Set the overall title for the figure based on the type of data
                fig.suptitle(f"Discriminator's view on {types.capitalize()} Data and the respective Attribution maps")
            
                # Store the figure with an appropriate name
                figs.append(fig)
                names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{types}_data", f"subset_{chunks.index(chunk) + 1}"))
        return figs, names


class Label_transition_plot(GenericPlot):
    def __init__(self, learner, label_index, label_names):
        super(Label_transition_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribute transition plots")
        
        self.label_index = label_index
        self.label_names = label_names
        self.start = 0.0
        self.end = 1.0
        self.steps = 10
    
    def consistency_check(self):
        return True

    def generate_attribute_transition(self, original_images, original_labels):
        transition_images = [original_images]  # Start with the original images
        step_size = 0.1

        # Loop to gradually change the attribute for each image
        for step in range(self.steps + 1):
            modified_attributes = original_labels.clone()

            for i in range(modified_attributes.size(0)):
                current_attr_value = modified_attributes[i, self.label_index]
                # Determine transition direction based on the initial attribute value
                if current_attr_value == 0:
                    # Example, transition from female to male
                    new_value = self.start + (step * step_size)
                else:
                    # Example transition from male to female
                    new_value = self.end - (step * step_size)

                modified_attributes[i, self.label_index] = new_value

            # Generate noise vector
            noise = torch.randn(original_images.size(0), self.learner.noise_dim, device=self.learner.device)
            
            # self.learner._load()  
            generator = self.learner.model.generator

            # Generate image with modified attributes
            generated_image = generator(noise, modified_attributes)
            transition_images.append(generated_image)
        return transition_images
    
    def plot(self):
        # Select a batch of images and labels
        total_real_images, total_labels = self.learner.data_storage.get_item("real_images"), self.learner.data_storage.get_item("labels")
        real_images, labels = total_real_images[-1][:5], total_labels[-1][:5]
    
        original_images, original_labels = real_images, labels
    
        generated_images = self.generate_attribute_transition(original_images, original_labels)
    
        # Plotting
        num_images = original_images.size(0)
        fig, axs = plt.subplots(num_images, self.steps + 3, figsize=(2 * (self.steps + 3), 2 * num_images))
    
        for j in range(num_images):
            original_label = original_labels[j, self.label_index].item()
            original_img = original_images[j].squeeze().detach().cpu().permute(1, 2, 0)

            original_img = (original_img + 1) / 2
            original_img = np.clip(original_img, 0, 1)
            
            # Set common headings for all row (0.0 to 1.0)
            if j == 0:
                for i in range(self.steps + 1):
                    transition_value = i * 0.1
                    axs[j, i + 1].set_title(f'{transition_value:.1f}')
    
            if original_label == 0:
                axs[j, 0].imshow(original_img)
                axs[j, 0].set_title('Original')
                axs[j, 0].axis("off")
                start_idx = 1
                
                # Turn off the axes for the last subplot on the right for label 0
                axs[j, -1].axis("off")
                    
                for i, img in enumerate(generated_images[1:]):
                    img = img[j].squeeze().detach().cpu().permute(1, 2, 0)

                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    
                    axs[j, start_idx + i].imshow(img)
                    axs[j, start_idx + i].axis('off')

            else:
                axs[j, -1].imshow(original_img)
                axs[j, -1].set_title('Original')
                axs[j, -1].axis("off")
                start_idx = self.steps + 1
                
                # Turn off the axes for the first subplot on the left for label 1
                axs[j, 0].axis("off")
                
                for i, img in enumerate(generated_images[1:]):
                    img = img[j].squeeze().detach().cpu().permute(1, 2, 0)

                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                        
                    axs[j, start_idx].imshow(img)
                    axs[j, start_idx].axis('off')
                    start_idx -= 1  # Move one subplot to the left
    
        fig.suptitle(f'Label Transition of {self.label_names}')
        plt.tight_layout()
    
        figs = [fig]
        names = [os.path.join("plots", "analysis_plots", "label_transition", f"transition_{self.label_names}")]
        return figs, names


class Probabilities_plot(GenericPlot):
    def __init__(self, learner):
        super(Probabilities_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of probabilities plots")

    def consistency_check(self):
        return True
    
    def values(self, types):
        image_list = self.learner.data_storage.get_item(f"{types}_images")
        labels_list = self.learner.data_storage.get_item("labels")
        
        imgs = image_list[-1][:40]
        labels = labels_list[-1][:40]
        
        return imgs, labels
    
    def plot(self):
        names = []
        figs = []
    
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        max_images_per_plot = 3  # Maximum number of images per plot
    
        for types in ["real", "fake"]:
            imgs, labels = self.values(types)
            
            classifier = Tsne_plot_classifier.load_classifier(self)
            outputs = classifier(imgs)
            preds = (outputs.detach() > 0.5).float()
            
            sigmoid = torch.nn.Sigmoid()
            sigmoid_outputs = sigmoid(outputs)
    
            # Process the data in chunks of max_images_per_plot
            num_chunks = len(imgs) // max_images_per_plot + (len(imgs) % max_images_per_plot > 0)
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_images_per_plot
                end_idx = start_idx + max_images_per_plot
                chunk_inputs = imgs[start_idx:end_idx]
                chunk_outputs = sigmoid_outputs[start_idx:end_idx]
                chunk_labels = labels[start_idx:end_idx]
                chunk_preds = preds[start_idx:end_idx]
    
                # Creating subplots for each chunk
                fig, axs = plt.subplots(2, len(chunk_inputs), figsize=(10 * len(chunk_inputs), 10), squeeze=False)
                fig.suptitle(f'Histogram of probabilities of labels\n{label_names}')
    
                for i in range(len(chunk_inputs)):
                    img = chunk_inputs[i].cpu().detach().permute(1, 2, 0).numpy()
                    
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    
                    label = chunk_labels[i].cpu().detach().numpy()
                    pred = chunk_preds[i].cpu().detach().numpy()
                    output_prob = chunk_outputs[i].cpu().detach().numpy()
    
                    axs[0, i].imshow(img)
                    axs[0, i].set_title(f"Image {start_idx + i + 1}")
                    axs[0, i].axis("off")
    
                    axs[1, i].bar(range(len(output_prob)), output_prob, color=['green' if pred[j] == label[j] else 'blue' for j in range(len(label))])
                    title_str = f"Label: {label}\nPred: {pred}"
                    axs[1, i].set_title(title_str, fontsize=12)
                    axs[1, i].set_xticks(range(len(output_prob)))
                    axs[1, i].set_xticklabels(label_names, rotation=45, ha="right")
                    axs[1, i].set_ylim((0, 1))
                    axs[1, i].set_yticks(np.arange(0, 1.1, 0.1))
                
                # Adjust legend for cases with fewer subplots
                correct_bar = plt.Rectangle((0,0),1,1,fc='green', edgecolor='none')
                incorrect_bar = plt.Rectangle((0,0),1,1,fc='blue', edgecolor='none')
                fig.legend([correct_bar, incorrect_bar], ['Correct', 'Incorrect'], loc='upper right', ncol=2)
    
                names.append(os.path.join("plots", "analysis_plots", "probabilities_plots_classifier", f"{types}_images", f"subset_{chunk_idx + 1}"))
                figs.append(fig)
                plt.close(fig)
    
        return figs, names
    

# Custom Dataset class to handle lists of tensors
class ImageTensorDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

# Other functions
def attributions(model, inputs):
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    # Initialize the Occlusion object
    occlusion = Occlusion(model)   
    return saliency, guided_backprop, input_x_gradient, deconv, occlusion

def attribution_maps(model, inputs, labels=None):
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model, inputs)
    
    if labels is not None:
        saliency_maps = saliency.attribute(inputs, additional_forward_args=labels)
        guided_backprop_maps = guided_backprop.attribute(inputs, additional_forward_args=labels)
        input_x_gradient_maps = input_x_gradient.attribute(inputs, additional_forward_args=labels)
        deconv_maps = deconv.attribute(inputs, additional_forward_args=labels)
        occlusion_maps = occlusion.attribute(inputs, additional_forward_args=labels, 
                                             sliding_window_shapes=(3, 10, 10),
                                             baselines = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(DEVICE))
    else:
        # For GAN, we do not pass the labels
        saliency_maps = saliency.attribute(inputs)
        guided_backprop_maps = guided_backprop.attribute(inputs)
        input_x_gradient_maps = input_x_gradient.attribute(inputs)
        deconv_maps = deconv.attribute(inputs)
        occlusion_maps = occlusion.attribute(inputs, 
                                             sliding_window_shapes=(3, 10, 10),
                                             baselines = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(DEVICE))
    return saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps   
