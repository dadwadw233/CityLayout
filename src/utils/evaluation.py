
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from tqdm.auto import tqdm


from utils.vis import OSMVisulizer

from utils.log import *


from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchmetrics.image import StructuralSimilarityIndexMeasure

import torch
import numpy as np
import os
import matplotlib.pyplot as plt



def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Evaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
        data_type="rgb",
        mapping=None,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler



        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        



        self.vis = OSMVisulizer(mapping)
        self.data_type = data_type


        if num_fid_samples < inception_block_idx:
            WARNING(
                f"num_fid_samples {num_fid_samples} is smaller than inception_block_idx {inception_block_idx}, "
                "this may cause error when computing FID, please set num_fid_samples larger than (or at least equal to) inception_block_idx"
            )

        # metrics for evaluate Image Similarity
        self.FID_calculator = FrechetInceptionDistance(feature=inception_block_idx, normalize=True).to(device)
        self.KID_calculator = KernelInceptionDistance(feature=inception_block_idx, normalize=True).to(device)
        self.MIFID_calculator = MemorizationInformedFrechetInceptionDistance(feature=inception_block_idx, normalize=True).to(device)

        # metrics for evaluate Image Quality
        self.IS_caluculator = InceptionScore(normalize=True).to(device)


        # MultoModal metrics
        self.CLIP_calculator = CLIPImageQualityAssessment(data_range=1.0).to(device)
        


        # predifine some prompts for CLIP
        color = 'green'
        self.prompt_pair = [
            ("a city region", "not a city region"),
            ("a city map include buildings and roads", "not a city map"),
            ("high quality city map", "low quality city map"),
            ("the background's color is "+color, "the background's color is not "+color),
            ("the city map real", "the city map fake"),

        ]

        # some matrics to evaluate conditional generation's consistency
        self.LPIPS_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device)
        self.SSIM_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


        self.evaluate_dict = {}

    def summary(self):
        pass

    def update_sampler(self, sampler):
        self.sampler = sampler

    @torch.inference_mode()
    def validation (self, condition=False):
        pass

    @torch.inference_mode()
    def sample_real_data(self, cond=None):
        if cond is not None:
            cond = cond.to(self.device)
        real_samples = next(self.dl)['layout']
        if cond is not None:
            real_samples = torch.cat([real_samples, cond], dim=1)
        if self.data_type == "one-hot":
            real_samples = self.vis.onehot_to_rgb(real_samples)
        
        real_samples = real_samples.to(self.device)
        return real_samples
        
        


    @torch.inference_mode()
    def evaluate(self):
        
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)

        self.print_fn(
            f"Stacking real sample..."
        )
        for batch in tqdm(batches):
            cond = None
            try:
                data = next(self.dl)
                real_samples = data['layout']
                if self.condition:
                    cond = data['condition'].to(self.device)
                else:
                    cond = None
                
            except StopIteration:
                break
            real_samples = real_samples.to(self.device)
            if cond is not None:
                real_samples = torch.cat([real_samples, cond], dim=1)
            if self.data_type == "one-hot":
                real_samples = self.vis.onehot_to_rgb(real_samples)
            
            real_samples = real_samples.to(self.device)
            self.FID_calculator.update(real_samples, True)
            self.KID_calculator.update(real_samples, True)

        self.print_fn(
            f"Stacking fake sample..."
        )
        for batch in tqdm(batches):
            if self.condition:
                cond = next(self.dl)['condition'].to(self.device)
            else:
                cond = None
            # do not cat cond to fake_samples, because the sample function has already done this
            fake_samples = self.sampler.sample(batch_size=batch, cond=cond)
            if self.data_type == "one-hot":
                fake_samples = self.vis.onehot_to_rgb(fake_samples)
            
            fake_samples = fake_samples.to(self.device)
            self.FID_calculator.update(fake_samples, False)
            self.KID_calculator.update(fake_samples, False)
            self.IS_caluculator.update(fake_samples)
        
            

        fid = self.FID_calculator.compute().item()
        kid_mean, kid_std = self.KID_calculator.compute()
        kid = kid_mean.item()
        
        is_mean, is_std = self.IS_caluculator.compute()
        is_score = is_mean.item()


        
        return fid, kid, is_score
    


class DataAnalyser:
    GREAT_COLOR_SET = plt.cm.get_cmap("tab20").colors

    def __init__(self, config=None):
        self._parse_config(config)
        self.init_folder()
        self.data_dict = self._init_data_dict()
        self.real_size = 0
        self.fake_size = 0

    def _parse_config(self, config):
        assert config is not None, "config must be provided"
        self.path = config["analyse"]["path"]
        self.layout_keys = config["data"]["custom_dict"]
        self.analyse_types = config["analyse"]["types"]
        self.threshold = config["analyse"]["threshold"]
        self.limit = config["analyse"]["evaluate_data_limit"]
        self.mapping = []

    def init_folder(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        else:
            for file in os.listdir(self.path):
                if file.endswith(".png") or file.endswith(".txt"):
                    os.remove(os.path.join(self.path, file))

    def _init_data_dict(self, fake=False):
        data_dict = {"real": {}, "fake": {}}
        
        for idx in self.layout_keys:
            for subgroup in self.layout_keys[idx]:
                subgroup_name = idx
                for key in subgroup:
                    subgroup_name += f"_{key}"
                    break

                self.mapping.append(subgroup_name)
                
                data_dict["real"][subgroup_name] = self._init_subgroup_data()
                data_dict["fake"][subgroup_name + "_fake"] = self._init_subgroup_data()
        
        for key in self.mapping.copy():
            self.mapping.append(key + "_fake")

        return data_dict

    def _init_subgroup_data(self):
        return {analyse_type: [] for analyse_type in self.analyse_types}

    @staticmethod
    def cal_overlap(data) -> np.float32:
        h, w = data.shape
        area = h * w
        overlap_rate = np.count_nonzero(data) / area
        return overlap_rate

    def _setup_plot(self, title=None, xlabel=None, ylabel=None, figsize=(10, 8)):
        """Prepare the plot with common settings."""
        plt.figure(figsize=figsize)
        plt.grid(True)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

    def _save_plot(self, filename):
        """Save the plot to the specified path."""
        file_path = os.path.join(self.path, filename)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _calculate_statistics(self, data):
        """Calculate statistics for the given data."""
        return {analyse_type: {"std": np.std(data[analyse_type]), "mean": np.mean(data[analyse_type])}
                for analyse_type in self.analyse_types}
    
    def _calculate_correlation(self, data1, data2):
        """Calculate correlation coefficient between two datasets."""
        try:
            corr_coef = np.corrcoef(data1, data2)[0, 1]
            return corr_coef if np.isfinite(corr_coef) else 0
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0

    def _flatten_data(self, data_dict, analyse_type):
        """Flatten the data for a specific analysis type."""
        return [data for values in data_dict.values() for data in values[analyse_type]]

    def _calculate_correlation_matrix(self, data_dict, analyse_type):
        """Calculate the correlation matrix for a specific analysis type."""
        sub_data = {key: values[analyse_type] for key, values in data_dict.items()}
        corr_matrix = np.zeros((len(sub_data), len(sub_data)))
        for i, (key1, values1) in enumerate(sub_data.items()):
            for j, (key2, values2) in enumerate(sub_data.items()):
                corr_matrix[i, j] = self._calculate_correlation(values1, values2)
        return corr_matrix

    def _cluster_data(self, data_dict):
        """Cluster the data based on correlation."""
        clusters = {}
        corr_matrix = {}
        for analyse_type in self.analyse_types:
            flattened_data = self._flatten_data(data_dict, analyse_type)
            corr_matrix[analyse_type] = self._calculate_correlation_matrix(data_dict, analyse_type)
            # ... Cluster the data based on the correlation matrix ...
        return clusters, corr_matrix

    def output_results_to_file(self, results, filename):
        """Write results to a text file."""
        with open(os.path.join(self.path, filename), "w") as file:
            for analyse_type in self.analyse_types:
                file.write(f"{analyse_type.upper()}\n")
                for key, values in results.items():
                    file.write(f"{key}: {values[analyse_type]}\n")
                file.write("\n")



    def plot_std_mean_all(self, data_dict, title="Mean and Standard Deviation for Each Category"):
        """Plot mean and standard deviation for all data categories."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Mean', ylabel='Standard Deviation')
            all_means = []
            all_stds = []
            for key, values in data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)])

            plt.xlim(left=min(all_means)*0.8, right=max(all_means)*1.2)
            plt.ylim(bottom=min(all_stds)*0.8, top=max(all_stds)*1.2)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_std_mean.png")
    
    def plot_hist_all(self, data_dict, title="Histogram for Each Category", bins=20, alpha=0.5):
        """Plot histograms for all data categories."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Value', ylabel='Frequency')
            for key, values in data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=bins, label=key, alpha=alpha, color=self.GREAT_COLOR_SET[self.mapping.index(key)])    

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_hist.png")


    def plot_curves_all(self, data_dict, title="Curves for Each Category"):
        """Plot curves for all data categories."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Epoch', ylabel='Value')
            for key, values in data_dict.items():
                plt.plot(values[analyse_type], label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)])

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_curves.png")
    

    def plot_correlation_matrix(self, corr_matrix, mapping, title="Correlation Matrix", cmap='coolwarm'):
        """Plot the correlation matrix."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=f"{title} - {analyse_type}", figsize=(10, 8))
            sub_corr_matrix = corr_matrix[analyse_type]
            plt.imshow(sub_corr_matrix, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(mapping)), mapping, rotation=90)
            plt.yticks(range(len(mapping)), mapping)
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_corr_matrix.png")

    def plot_fake_and_real_std_mean(self, data_dict, fake_data_dict, title="Mean and Standard Deviation for Each Category"):
        '''plot the contrast between fake and real data'''
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Mean', ylabel='Standard Deviation')
            all_means = []
            all_stds = []
            for key, values in data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)])

            for key, values in fake_data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)//2], marker='x')

            plt.xlim(left=min(all_means)*0.8, right=max(all_means)*1.2)
            plt.ylim(bottom=min(all_stds)*0.8, top=max(all_stds)*1.2)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_std_mean_fake_and_real.png")
    
    def plot_fake_and_real_hist(self, data_dict, fake_data_dict, title="Histogram for Each Category"):
        """Plot histograms for all data categories."""
        

        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Value', ylabel='Frequency')
            for key, values in data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=20, label=key, alpha=0.5, color=self.GREAT_COLOR_SET[self.mapping.index(key)])    

            for key, values in fake_data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=20, label=key, alpha=0.5, color=self.GREAT_COLOR_SET[self.mapping.index(key)//2], histtype='step', linestyle='dashed')    

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_hist_fake_and_real.png")

    def plot_fake_and_real_correlation_matrix(self, corr_matrix, mapping, title="Correlation Matrix"):
        """Plot the correlation matrix."""
        for analyse_type in self.analyse_types:
            sub_corr_matrix = corr_matrix[analyse_type]
            self._setup_plot(title=title, figsize=(10, 8))
            plt.imshow(sub_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(mapping)), mapping, rotation=90)
            plt.yticks(range(len(mapping)), mapping)
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_corr_matrix_fake_and_real.png")


    def analyse(self, fake: bool = False):
        """Main method to run the analysis."""
        assert self.real_size != 0 if not fake else self.fake_size != 0, "please add data first"
        # Perform calculations
        data_for_analysis = self.data_dict["fake"] if fake else self.data_dict["real"]
        statistics = {key: self._calculate_statistics(values) for key, values in data_for_analysis.items()}
        clusters, corr_matrix = self._cluster_data(data_for_analysis)
        
        # Plot results
        self.plot_std_mean_all(data_for_analysis)
        self.plot_hist_all(data_for_analysis)
        self.plot_curves_all(data_for_analysis)

        self.plot_correlation_matrix(corr_matrix, self.mapping[:self.mapping.__len__()//2])

        # Output results
        self.output_results_to_file(statistics, "statistics.txt")
        self.output_results_to_file(clusters, "clusters.txt")



    def contrast_analyse(self):
        """analyse the contrast between fake and real data"""
        assert self.real_size != 0 and self.fake_size != 0, "please add data first"


        evaluation_data_size = self.real_size if self.real_size < self.fake_size else self.fake_size
        print(f"evaluation data size: {evaluation_data_size}", f"\nbecause real data size: {self.real_size} and fake data size: {self.fake_size}")

        real_for_analysis = {}
        fake_for_analysis = {}
        for key, values in self.data_dict["real"].items():
            real_for_analysis[key] = {analyse_type: values[analyse_type][:evaluation_data_size] for analyse_type in self.analyse_types}
        for key, values in self.data_dict["fake"].items():
            fake_for_analysis[key] = {analyse_type: values[analyse_type][:evaluation_data_size] for analyse_type in self.analyse_types}


        statistics = {key: self._calculate_statistics(values) for key, values in real_for_analysis.items()}
        fake_statistics = {key: self._calculate_statistics(values) for key, values in fake_for_analysis.items()}

        uni_dict = {}
        for key, values in real_for_analysis.items():
            uni_dict[key] = values
        for key, values in fake_for_analysis.items():
            uni_dict[key] = values
        clusters, corr_matrix = self._cluster_data(uni_dict)
        

        self.plot_fake_and_real_hist(real_for_analysis, fake_for_analysis, title="Histogram for Each Category")  
        self.plot_fake_and_real_std_mean(real_for_analysis, fake_for_analysis, title="Mean and Standard Deviation for Each Category")  
        
        self.plot_fake_and_real_correlation_matrix(corr_matrix, self.mapping, title="Correlation Matrix")

        

        # Output results
        self.output_results_to_file(statistics, "real_statistics.txt")
        self.output_results_to_file(fake_statistics, "fake_statistics.txt")
        self.output_results_to_file(clusters, "uni_clusters.txt")

        
    # some helper functions 👇
    # some emoji for fun
    # 👨 👈 🤡
    # 👇 🚮
    # copilot 👉 🐂🍺


    def get_data_size(self) -> tuple:

        return self.real_size, self.fake_size
    

    def release_data (self, fake: bool = False):
        if fake:
            self.data_dict["fake"] = self._init_data_dict(fake=True)
        else:
            self.data_dict["real"] = self._init_data_dict(fake=False)
            

    
    def add_data(self, data: torch.tensor, fake: bool = False):
        if data.device != "cpu":
            data = data.cpu()
        
        b, c, h, w = data.shape

        if fake:
            self.fake_size += b
        else:
            self.real_size += b

        now_data_size = self.fake_size if fake else self.real_size
        if now_data_size > self.limit:
            return


        for idx in range(b):
            for c_id in range(c):
                if "overlap" in self.analyse_types:
                    if fake:
                        self.data_dict["fake"][self.mapping[c_id] + "_fake"]["overlap"].append(self.cal_overlap(data[idx, c_id, :, :]))
                    else:
                        self.data_dict["real"][self.mapping[c_id]]["overlap"].append(self.cal_overlap(data[idx, c_id, :, :]))