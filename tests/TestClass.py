import yaml


class TestClass():   
    def __init__(self, config_file : dict = "config.yaml"):
        try:
            with open(config_file, 'r') as yaml_file:
                        self.config = yaml.safe_load(yaml_file)
            print("Class initialized")
            
        except FileNotFoundError:
            raise ValueError("Configuration file not found.")
            
        # Define a dictionary of default values
        default_values = {'architecture': "unet", "n_patches" : 1000, 'radius' : 40}
                
        # Update the loaded configuration with missing keys and default values
        self.config = self.update_with_defaults(self.config, default_values)

        # Set attributes based on configuration keys
        for key, value in self.config.items():
            setattr(self, key, value)
            
            
    def update_with_defaults(self, config, defaults):
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = self.update_with_defaults(config[key], value)
        return config
    
        
    def _get_config(self):
      return self.config
  
    def _get_architecture(self):
        print(self.architecture)
        
    def _set_architecture(self, architecture):
        self.architecture = architecture
        
    
    def do_stuff(self, **kwargs):
        if 'architecture' in kwargs:
            self._set_architecture(kwargs['architecture'])        
        self._get_architecture()
        
    def do_stuff2(self, *args):
        for arg in args:
            if arg.startswith("architecture="):
                _, architecture = arg.split("=")
                self.set_architecture(architecture)       
        self._get_architecture()
        
        
        
        

test = TestClass(config_file = "config.yaml")
test.do_stuff(architecture = "unet2")        