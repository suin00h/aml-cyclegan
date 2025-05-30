import yaml

class TrainConfig:
    def __init__(self, yaml_path):
        self.config_yaml = self.load_config(yaml_path)
        
        # Set train/test mode
        self.mode = 'train'

        # Set train parameters
        training_args = self.config_yaml.get("training_args", {})
        self.lr = float(training_args.get("lr", 0.0002))
        self.lambda_x = float(training_args.get("lambda_x", 10))
        self.lambda_y = float(training_args.get("lambda_y", 10))
        self.lambda_idt = float(training_args.get("lambda_idt", 0.5))
        self.epoch_cnt = int(training_args.get("epoch_cnt", 1))
        self.n_epochs = int(training_args.get("n_epochs", 100))
        self.n_epochs_decay = int(training_args.get("n_epochs_decay", 100))
        self.batch_size = int(training_args.get("batch_size", 1))
        self.save_epochs = int(training_args.get("save_epochs", 5))
        self.max_buffer = int(training_args.get("max_buffer", 5))

        # Set transform options
        transform = self.config_yaml.get("transform", {})
        self.resize = transform.get("resize", True)
        self.grayscale = transform.get("grayscale", False)
        self.normalize = transform.get("normalize", True)
        self.size = transform.get("size", [256, 256])

        # Set dataset path
        dataset = self.config_yaml.get("dataset", {})
        self.dataname = dataset.get("name")
        self.datapath = dataset.get("path")
        self.direction = dataset.get("direction")
        self.input_nc = dataset.get("A_channel")
        self.output_nc = dataset.get("B_channel")
        self.save_dir = dataset.get("save_dir")

    def update_recursive(self, dict1, dict2):
        """
        update two config dictionaries recursively
        Args:
            dict1: (dict), first dictionary to be updated
            dictw: (dict), second dictionary which entries should be used

        Returns:

        """
        for k, v in dict2.items():
            if k not in dict1:
                dict1[k] = dict()
            if isinstance(v, dict):
                self.update_recursive(dict1[k], v)
            else:
                dict1[k] = v

    def load_config(self, path, default_path=None):
        """
        Load config file
        Args:
            path: (str), path to config file
            default_path: (str, optional), whether to use default path.

        Returns:
            cfg: (dict), config dict

        """
        # load configuration from file itself
        with open(path, 'r' ) as f:
            cfg_special = yaml.full_load(f)

        # check if we should inherit from a config
        inherit_from = cfg_special.get('inherit_from')

        # if yes, load this config first as default
        # if no, use the default path
        if inherit_from is not None:
            cfg = self.load_config(inherit_from, default_path)
        elif default_path is not None:
            with open(default_path, 'r') as f:
                cfg = yaml.full_load(f)
        else:
            cfg = dict()

        # include main configuration
        self.update_recursive(cfg, cfg_special)

        return cfg