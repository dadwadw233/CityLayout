
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
import wandb


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
        mapping=None,
        config=None,
    ):
        self.batch_size = batch_size
        
        self.device = device
        self.channels = channels

        self.dl = dl
        self.sampler = sampler

        self.num_fid_samples = config['num_fid_samples']
        self.inception_block_idx = config['inception_block_idx']
        self.config = config
        
        self.metrics_list = config['metrics_list']
        self.data_type = config['data_type']


  


        self.print_fn = print if accelerator is None else accelerator.print
        assert self.inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        



        self.vis = OSMVisulizer(mapping)
        


        if self.num_fid_samples < self.inception_block_idx:
            WARNING(
                f"num_fid_samples {self.num_fid_samples} is smaller than inception_block_idx {self.inception_block_idx}, "
                "this may cause error when computing FID, please set num_fid_samples larger than (or at least equal to) inception_block_idx"
            )

        # metrics for evaluate Image Similarity
        self.FID_calculator = FrechetInceptionDistance(feature=self.inception_block_idx, normalize=True).to(device)
        self.KID_calculator = KernelInceptionDistance(feature=self.inception_block_idx, normalize=True, subset_size=self.num_fid_samples//4).to(device)
        self.MIFID_calculator = MemorizationInformedFrechetInceptionDistance(feature=self.inception_block_idx, normalize=True).to(device)

        # metrics for evaluate Image Quality
        self.IS_caluculator = InceptionScore(normalize=True).to(device)


        
        


        # predifine some prompts for CLIP
        
        self.prompt_pair = (
            ("a city region", "not a city region"),
            ("a city map include buildings and roads", "a city map without buildings and roads"),
            ("high quality city map", "low quality city map"),
            ("the background's color is black", "the background's color is not black"),
            ("a real city map", "a fake city map")
        )
        # MultoModal metrics
        self.CLIP_calculator = CLIPImageQualityAssessment(data_range=1.0, prompts=self.prompt_pair).to(device)

        # some matrics to evaluate conditional generation's consistency
        self.LPIPS_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device)
        self.SSIM_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # data satistics evaluator
        self.data_analyser = DataAnalyser(config=config)

        self.evaluate_dict = {}

        self.condition = False

    def get_evaluation_dict(self):
        return self.evaluate_dict
    

    def set_condtion(self, condition):
        self.condition = condition

    def recursive(self, d) -> str:
        # help get_summary function to convert dict to string recursively
        summary_str = ""
        for k, v in d.items():
            if isinstance(v, dict):
                summary_str += self.recursive(v)
            else:
                summary_str += f"{k}: {v}\n"
        return summary_str

    def summary(self) -> str:
        summary_str = ""
        summary_str += "Begin Evaluation Summary:\n"
        summary_str += "----------------------------------------\n"
        summary_str += "key: value\n"
        summary_str += "----------------------------------------\n"
        for k, v in self.config.items():
            # consider the case that v is a dict recursively
            if isinstance(v, dict):
                summary_str += self.recursive(v)
            else:
                summary_str += f"{k}: {v}\n"

        summary_str += "----------------------------------------\n"
        summary_str += "End Evaluation Summary\n"

        return summary_str

        

    def update_sampler(self, sampler):
        self.sampler = sampler

    @torch.inference_mode()
    def validation (self, condition=False,  path=None):
        # init evaluate dict
        self.evaluate_dict = {}
        self.evaluate_dict['CLIP'] = {}
        for prompt in self.prompt_pair:
            prompt_str = prompt[0] + " vs " + prompt[1]
            self.evaluate_dict['CLIP'][prompt_str] = {'fake': [], 'real': []}

        self.set_condtion(condition)

        if not self.image_evaluation(path=path):
            ERROR("Error occured when evaluating images")
            return False
        else:
            INFO("Evaluation finished")
            return True

    @torch.inference_mode()
    def sample_real_data(self):
        
        real_samples = next(self.dl)['layout']
        self.data_analyser.add_data(real_samples, fake=False)
        
        if self.data_type == "one-hot":
            real_samples = self.vis.onehot_to_rgb(real_samples)
        
        real_samples = real_samples.to(self.device)
        return real_samples
    
    @torch.inference_mode()
    def sample_fake_data(self, cond=None):
        if cond is not None:
            cond = cond.to(self.device)
            fake_samples = self.sampler.sample(batch_size=self.batch_size, cond=cond)
            fake_samples = fake_samples[:, :self.channels, :, :]
        else:
            fake_samples = self.sampler.sample(batch_size=self.batch_size, cond=None)
        self.data_analyser.add_data(fake_samples, fake=True)
        if self.data_type == "one-hot":
            fake_samples = self.vis.onehot_to_rgb(fake_samples)
        
        fake_samples = fake_samples.to(self.device)
        return fake_samples
        
    

    @torch.inference_mode()
    def multimodal_evaluation(self, real, fake):
        # init evaluate dict
        
        try:
            real_score = self.CLIP_calculator(real)
            fake_score = self.CLIP_calculator(fake)


            keys = real_score.keys()
            for pk, clip_k in zip(self.prompt_pair, keys):
                
                prompt_key = pk[0] + " vs " + pk[1]
                self.evaluate_dict['CLIP'][prompt_key]['fake'].append(fake_score[clip_k].mean().item())
                self.evaluate_dict['CLIP'][prompt_key]['real'].append(real_score[clip_k].mean().item())
                
            return True
        except Exception as e:
            ERROR(f"Error when calculating CLIP: {e}")
            return False
        



    @torch.inference_mode()
    def image_evaluation(self, path=None):

        try:    
        
            self.sampler.eval()

            batches = num_to_groups(self.num_fid_samples, self.batch_size)

            INFO(f"Start evaluating images")
            
            for batch in tqdm(batches, leave=False):
                real_samples = self.sample_real_data()
                
                real_samples = real_samples.to(self.device)
                self.FID_calculator.update(real_samples, True)
                self.KID_calculator.update(real_samples, True)
                self.MIFID_calculator.update(real_samples, True)

            
            
                if self.condition:
                    cond = next(self.dl)['layout'].to(self.device)
                else:
                    cond = None

                fake_samples = self.sample_fake_data(cond=cond)
                self.FID_calculator.update(fake_samples, False)
                self.KID_calculator.update(fake_samples, False)
                self.MIFID_calculator.update(fake_samples, False)
                self.IS_caluculator.update(fake_samples)
                if 'clip' in self.metrics_list:
                    self.multimodal_evaluation(real_samples, fake_samples)

            # sample some real data and fake data to show by wandb
            if not self.condition:
                real_samples = self.sample_real_data()
                fake_samples = self.sample_fake_data(cond = None)
                real_samples = real_samples[0].permute(1, 2, 0).cpu().numpy()
                fake_samples = fake_samples[0].permute(1, 2, 0).cpu().numpy()
                wandb.log({"real": [wandb.Image(real_samples, caption="real")]}, commit=False)
                wandb.log({"fake": [wandb.Image(fake_samples, caption="fake")]}, commit=False)
            
                

            if 'fid' in self.metrics_list:
                self.evaluate_dict['FID'] = self.FID_calculator.compute().item()
            if 'kid' in self.metrics_list:
                mean, std = self.KID_calculator.compute()
                self.evaluate_dict['KID'] = mean.item()
                self.evaluate_dict['KID_std'] = std.item()
            if 'mifid' in self.metrics_list:
                self.evaluate_dict['MIFID'] = self.MIFID_calculator.compute().item()
            if 'is' in self.metrics_list:
                mean, std = self.IS_caluculator.compute()
                self.evaluate_dict['IS'] = mean.item()
                self.evaluate_dict['IS_std'] = std.item()
            if 'clip' in self.metrics_list:
                # get mean score for each prompt
                for prompt in self.prompt_pair:
                    prompt_str = prompt[0] + " vs " + prompt[1]
                    self.evaluate_dict['CLIP'][prompt_str]['fake'] = np.mean(self.evaluate_dict['CLIP'][prompt_str]['fake'])
                    self.evaluate_dict['CLIP'][prompt_str]['real'] = np.mean(self.evaluate_dict['CLIP'][prompt_str]['real'])
            if 'data_analysis' in self.metrics_list:
                self.data_analyser.contrast_analyse(path, self.condition)
            self.data_analyser.release_data()
            

            return True
        
        except Exception as e:
            ERROR(f"Error when calculating FID and KID: {e}")
            self.evaluate_dict['FID'] = None
            self.evaluate_dict['KID'] = None
            self.evaluate_dict['MIFID'] = None
            self.evaluate_dict['IS'] = None
            self.evaluate_dict['CLIP'] = None

            return False
    


class DataAnalyser:
    GREAT_COLOR_SET = plt.cm.get_cmap("tab20").colors

    def __init__(self, config=None):
        self._parse_config(config)
        self.init_folder()
        self.data_dict = self._init_data_dict()
        self.real_size = 0
        self.fake_size = 0
        self.condition = None
        

    def _parse_config(self, config):
        assert config is not None, "config must be provided"
        if "path" not in config:
            self.path = None
        else:
            self.path = config["path"]
        self.layout_keys = config["custom_dict"]
        self.analyse_types = config["types"]
        self.threshold = config["threshold"]
        self.limit = config["evaluate_data_limit"]

        self.mapping = []
        

    def init_folder(self):
        if self.path is None:
            WARNING("path is not provided, plz set path param while using analyse function, or default path will be used: ./analysis")
            return
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        else:
            for file in os.listdir(self.path):
                if file.endswith(".png") or file.endswith(".txt"):
                    os.remove(os.path.join(self.path, file))

    def _init_data_dict(self):
        data_dict = {"real": {}, "fake": {}}
        self.mapping = []
        
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
    def cal_overlap(self, data) -> np.float32: # todo use the threshold to calculate the overlap
        h, w = data.shape
        area = h * w
        region = np.where(data > self.threshold, 1, 0)
        overlap_area = np.sum(region)
        overlap_rate = overlap_area / area
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
        # upload to wandb
        if self.condition:
            wandb.log({filename+'_cond': wandb.Image(file_path)}, commit=False)
        else:
            wandb.log({filename: wandb.Image(file_path)}, commit=False)
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

            length = data_dict.__len__()

            for key, values in fake_data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key) - length], marker='x')

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
            
            length = data_dict.__len__()

            for key, values in fake_data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=20, label=key, alpha=0.5, color=self.GREAT_COLOR_SET[self.mapping.index(key) - length], histtype='step', linestyle='dashed')    

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

    def set_condtion(self, condition):
        self.condition = condition

    def contrast_analyse(self, path=None, condition=None):
        """analyse the contrast between fake and real data"""
        assert self.real_size != 0 and self.fake_size != 0, "please add data first"
        if condition is not None:
            self.set_condtion(condition)
        if path is not None:
            self.path = path
            self.init_folder()
            flag = True
        elif self.path is None:
            self.path = os.path.join(os.getcwd(), "analysis")
            self.init_folder()
            flag = True
            WARNING("path is not provided, use default path: ./analysis")


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

        if flag:
            self.path = None

        
    # some helper functions 👇
    # some emoji for fun
    # 👨 👈 🤡
    # 👇 🚮
    # copilot 👉 🐂🍺


    def get_data_size(self) -> tuple:

        return self.real_size, self.fake_size
    

    def release_data (self):
        self.data_dict = self._init_data_dict()
        self.real_size = 0
        self.fake_size = 0
            

    
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