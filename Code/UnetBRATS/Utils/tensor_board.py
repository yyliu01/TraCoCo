import wandb
import torch
import torchvision
import os
import matplotlib.pyplot as plt
import imageio
import logging
import os
import zipfile
from pathlib import Path
from typing import Union
from google.cloud import storage
from tqdm import tqdm


class Tensorboard:
    def __init__(self, config):
        # os.environ['WANDB_API_KEY'] = config['wandb_key']
        os.system("wandb {}".format("online" if config.use_wandb else "disabled"))
        config.run_name = "traCoCo({}, {}-label, unsup_weight={})[{}]".format(str(config.dataset),
                                                                              str(config.labeled_num),
                                                                              str(config.unsup_weight),
                                                                              str(config.dataset))
        self.tensor_board = wandb.init(project=config.project_name,
                                       name=config.run_name,
                                       config=config)
        self.ckpt_root = 'saved'
        self.ckpt_path = os.path.join(self.ckpt_root, config.run_name)
        self.visual_root_path = os.path.join(self.ckpt_path, 'history_images')
        self.visual_results_root = os.path.join(self.visual_root_path, 'results')
        self._safe_mkdir(self.ckpt_root)
        self._safe_mkdir(self.ckpt_path)
        self._safe_mkdir(self.visual_root_path)
        self._safe_mkdir(self.visual_results_root)
        self._safe_mkdir(self.ckpt_root, config.run_name)

    def upload_wandb_info(self, info_dict, current_step=0):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info]})
        return

    def upload_wandb_table(self, info_dict):
        table = wandb.Table(data=info_dict['entire_prob_boundary'],
                            columns=["boundary", "rate"])
        wandb.log({"pass_in_each_boundary": wandb.plot.bar(table, "boundary", "rate",
                                                           title="PASS_RATE Bar Chart")})
        table = wandb.Table(data=info_dict['max_prob_boundary'],
                            columns=["boundary", "rate"])

        wandb.log({"max_prob_in_each_boundary": wandb.plot.bar(table, "boundary", "rate",
                                                               title="MAX_Prob Bar Chart")})

    def produce_2d_slice(self, image, label, pred):
        image = image[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = torchvision.utils.make_grid(image, 5, normalize=True)
        wandb.log({"volume_slice": [wandb.Image(grid_image,
                                                caption="x")]})

        outputs_soft = torch.softmax(pred, 1)
        image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_image = torchvision.utils.make_grid(image, 5, normalize=False)
        wandb.log({"output_slice": [wandb.Image(grid_image,
                                                caption="y_tilde")]})

        image = label[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1).float()
        grid_image = torchvision.utils.make_grid(image, 5, normalize=False)
        wandb.log({"label_slice": [wandb.Image(grid_image,
                                               caption="y")]})
        return

    def produce_3d_gif(self, current_epoch,
                       result, name="?", color="red"):
        current_path = os.path.join(self.visual_results_root, str(current_epoch))
        self._safe_mkdir(parent_path=self.visual_results_root, build_path=str(current_epoch))
        self._safe_mkdir(parent_path=current_path, build_path=name)
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(result, facecolors=color)
        ax.grid(False)
        plt.axis('off')
        for ii in [0, 90, 180, 270]:
            ax.view_init(elev=10., azim=ii)
            plt.savefig(os.path.join(current_path, name, "{}.png".format(ii)),
                        bbox_inches='tight', pad_inches=0, dpi=200)
        images = []
        for filename in sorted(sorted(os.listdir(os.path.join(current_path, name)),
                                      key=lambda x: int(x.split('.')[0]))):
            images.append(imageio.imread(os.path.join(current_path, name, filename)))
        imageio.mimsave(current_path + '/{}.gif'.format(name), images, duration=1)
        wandb.log({"{}_gif".format(name): wandb.Video(current_path + '/{}.gif'.format(name),
                                                      fps=4, format="gif")})
        plt.clf()
        return

    @staticmethod
    def _safe_mkdir(parent_path, build_path=None):
        if build_path is None:
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)
        else:
            if not os.path.exists(os.path.join(parent_path, build_path)):
                os.mkdir(os.path.join(parent_path, build_path))
        return

    def save_ckpt(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, name))
        return

    @staticmethod
    def get_bucket(bucket_namespace: str, bucket_name: str):
        client = storage.Client(project=bucket_namespace)
        bucket = client.get_bucket(bucket_name)
        return bucket

    # @staticmethod
    def upload_ckpt(self, local_path: str, prefix: str, checkpoint_filepath: Union[Path, str]):
        """Upload a model checkpoint to the specified bucket in GCS."""
        bucket_namespace = "aiml-carneiro-research"
        bucket_name = "aiml-carneiro-research-data"
        bucket_prefix = prefix
        src_path = f"{local_path}/{checkpoint_filepath}"
        dst_path = f"{bucket_prefix}/{checkpoint_filepath}"
        print('Uploading {} => {}'.format(src_path, dst_path))
        bucket = self.get_bucket(bucket_namespace, bucket_name)
        blob = bucket.blob(dst_path)
        blob.upload_from_filename(src_path)
        print('finish uploading.')
