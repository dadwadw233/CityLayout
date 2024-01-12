import os
import yaml


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
                raise ValueError("Config file does not exist!")

    def parse(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config
    
    def print_dict(self, d, indent=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    print("  " * indent + str(k) + ":")
                    self.print_dict(v, indent + 1)
                else:
                    print("  " * indent + str(k) + ": " + str(v))
    
    def get_summary(self):
        
        # print config summary
        print("Config Summary:")
        # if value type is dict, recursively print
        for k, v in self.config.items():
            if isinstance(v, dict):
                print("  " + str(k) + ":")
                self.print_dict(v, 2)
            else:
                print("  " + str(k) + ": " + str(v))
        
        

    def register_config(self, path):
        self.config_path = path
        if os.path.exists(path):
            self.config = self.parse()
        else:
            raise ValueError("Config file does not exist!")
    
    def get_config_by_name(self, name):
        if self.config is None:
            raise ValueError("Config is not registered!")
        elif name not in self.config.keys():
            raise ValueError("Config name is not in the config file!")
        return self.config[name]
    
    def get_config_all(self):
        if self.config is None:
            raise ValueError("Config is not registered!")
        return self.config
    
    def get_config_path(self):
        if self.config_path is None:
            raise ValueError("Config path is not registered!")
        return self.config_path
    



        