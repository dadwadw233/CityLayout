import os
import yaml
from utils.log import *


class ConfigParser:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = None
        if config_path is not None:
            if os.path.exists(config_path):
                self.config = self.parse()
            else:
                self.config = None
                self.config_path = None 
                WARNING(f"load config failed, config file {config_path} does not exist!")
                
        else:
            WARNING("config path is not provided! Please use set_config() to provide config dict or register_config() to provide config path")

    def set_config(self, config: dict):

        self.config = config

    def convert_none(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self.convert_none(v)
            else:
                if v == "None":
                    d[k] = None
        return d

    def parse(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # if config have some value is "None" string, convert it to None type
        config = self.convert_none(config)
        return config
    
    def recursive(self, d) -> str:
        # help get_summary function to convert dict to string recursively
        summary_str = ""
        for k, v in d.items():
            if isinstance(v, dict):
                summary_str += self.recursive(v)
            else:
                summary_str += f"{k}: {v}\n"
        return summary_str

    
    def get_summary(self):
        
        # compact config detail to a string witch will organize like a table
        summary_str_table_like = "Begin Config Summary:\n"
        summary_str_table_like += "----------------------------------------\n"
        summary_str_table_like += "key: value\n"
        summary_str_table_like += "----------------------------------------\n"
        for k, v in self.config.items():
            # consider the case that v is a dict recursively
            if isinstance(v, dict):
                summary_str_table_like += self.recursive(v)
            else:
                summary_str_table_like += f"{k}: {v}\n"
        
        summary_str_table_like += "----------------------------------------\n"
        summary_str_table_like += "End Config Summary\n"
            
        return summary_str_table_like
        
    
    def recursive_replace_by_key(self, d, key, value):
        # help replace_by_key function to replace value by key recursively
        for k, v in d.items():
            if isinstance(v, dict):
                self.recursive_replace_by_key(v, key, value)
            else:
                if k == key:
                    d[k] = value
        return d
    
    def renew_by_sweep_config(self, sweep_config):
        # renew config by sweep_config
        # sweep_config is a dict, which contain the key-value pairs that need to be replaced
        # for example, sweep_config = {"lr": 0.001, "beta_schedule": "sigmoid"}
        # then the config will be renewed by replacing the value of key "lr" to 0.001 and the value of key "beta_schedule" to "sigmoid"
        # if the key is not in config, then this key-value pair will be added to config
        # if the sweep_config is None, then the config will not be changed
        
        if sweep_config is None:
            return
        for k, v in sweep_config.items():
            INFO(f"renew config by sweep config: {k}: {v}")
            self.config = self.recursive_replace_by_key(self.config, k, v)
        
        

    def register_config(self, path):
        self.config_path = path
        if os.path.exists(path):
            self.config = self.parse()
        else:
            raise ValueError("Config file does not exist!")
    
    def get_config_by_name(self, name):
        if self.config is None:
            ERROR("Config is not registered!")
            raise ValueError("Config is not registered!")
        elif name not in self.config.keys():
            ERROR(f"Config name {name} is not in the config file!")
            raise ValueError(f"Config name {name} is not in the config file!")
        return self.config[name]
    
    def get_config_all(self):
        if self.config is None:
            raise ValueError("Config is not registered!")
        return self.config
    
    def get_config_path(self):
        if self.config_path is None:
            raise ValueError("Config path is not registered!")
        return self.config_path
    



        