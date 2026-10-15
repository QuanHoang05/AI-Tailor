import os
import os.path as osp
import sys
import datetime
import shutil
from mmcv import Config as MMConfig

class Config:
    def __init__(self):
        # Khởi tạo sẵn các đường dẫn cốt lõi ngay khi tạo đối tượng cfg = Config()
        # Tránh lỗi AttributeError: 'Config' object has no attribute 'human_model_path'
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.abspath(osp.join(self.cur_dir, '..'))
        self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

    def get_config_fromfile(self, config_path):
        self.config_path = config_path
        cfg = MMConfig.fromfile(self.config_path)
        self.__dict__.update(dict(cfg))

        # Cập nhật lại đường dẫn sau khi nạp config
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.abspath(osp.join(self.cur_dir, '..'))
        self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

        # Đảm bảo đường dẫn ViTPose encoder luôn đúng dù gọi từ thư mục gốc hay thư mục main/
        if hasattr(self, 'encoder_config_file') and not osp.exists(self.encoder_config_file):
            self.encoder_config_file = osp.join(self.cur_dir, self.encoder_config_file)
        if hasattr(self, 'encoder_pretrained_model_path') and not osp.exists(self.encoder_pretrained_model_path):
            self.encoder_pretrained_model_path = osp.abspath(osp.join(self.cur_dir, self.encoder_pretrained_model_path))

        sys.path.insert(0, osp.join(self.root_dir, 'common'))
        from utils.dir import add_pypath
        add_pypath(osp.join(self.data_dir))
        data_root = osp.join(self.root_dir, 'data')
        if osp.exists(data_root):
            for dataset in os.listdir(data_root):
                if dataset not in ['humandata.py', '__pycache__', 'dataset.py']:
                    add_pypath(osp.join(data_root, dataset))
            add_pypath(data_root)
        add_pypath(self.data_dir)

    def prepare_dirs(self, exp_name):
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = osp.join(self.root_dir, f'{exp_name}_{time_str}')
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.code_dir = osp.join(self.output_dir, 'code')
        self.result_dir = osp.join(self.output_dir, 'result')

        from utils.dir import make_folder
        make_folder(self.model_dir)
        make_folder(self.vis_dir)
        make_folder(self.log_dir)
        make_folder(self.code_dir)
        make_folder(self.result_dir)

        copy_files = ['common/base.py', 'common/nets', 'main/SMPLer_X.py']
        for file in copy_files:
            src_f = osp.join(self.root_dir, file)
            if osp.exists(src_f):
                try:
                    dst_f = osp.join(self.code_dir, osp.basename(file))
                    if osp.isdir(src_f):
                        shutil.copytree(src_f, dst_f, dirs_exist_ok=True)
                    else:
                        shutil.copy(src_f, dst_f)
                except Exception:
                    pass

    def update_test_config(self, testset, agora_benchmark, shapy_eval_split, pretrained_model_path, use_cache,
                           eval_on_train=False, vis=False):
        self.testset = testset
        self.agora_benchmark = agora_benchmark
        self.pretrained_model_path = pretrained_model_path
        self.shapy_eval_split = shapy_eval_split
        self.use_cache = use_cache
        self.eval_on_train = eval_on_train
        self.vis = vis

    def update_config(self, num_gpus, exp_name):
        self.num_gpus = num_gpus
        self.exp_name = exp_name
        self.prepare_dirs(self.exp_name)
        cfg_save = MMConfig(self.__dict__)
        cfg_save.dump(osp.join(self.code_dir, 'config_base.py'))

cfg = Config()
