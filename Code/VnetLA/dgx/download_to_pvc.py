import os
import zipfile
from pathlib import Path
from typing import Union
from google.cloud import storage


bucket_namespace = "your name space"
bucket_name = "your bucket name"


def get_bucket(bucket_namespace: str, bucket_name: str):
    client = storage.Client(project=bucket_namespace)
    bucket = client.get_bucket(bucket_name)
    return bucket


def download_la_unzip(data_dir: str, prefix, pvc=False):
    bucket_prefix = 'la.zip'
    dst_folder = Path(data_dir)
    root_dir = '/pvc/' if pvc else './'
    print('Destination Foder list ==> {}'.format(dst_folder))
    if dst_folder.exists():
        print('Skipping download as data dir already exists')
    else:
        print('searching blob ...')
        bucket = get_bucket('your bucket namespace', 'your bucket name')
        blob = bucket.blob(prefix+bucket_prefix)
        print('downloading ...')
        with open(Path(root_dir)/bucket_prefix, 'wb') as sink:
            blob.download_to_file(sink)
        print('unziping the {} ...'.format(bucket_prefix))
        with zipfile.ZipFile('{}/la.zip'.format(root_dir), 'r') as zip_ref:
            zip_ref.extractall(root_dir)
    print(os.listdir(root_dir))


def upload_checkpoint(local_path: str, prefix: str, checkpoint_filepath: Union[Path, str]):
    """Upload a model checkpoint to the specified bucket in GCS."""
    bucket_prefix = prefix
    src_path = f"{local_path}/{checkpoint_filepath}"
    dst_path = f"{bucket_prefix}/{checkpoint_filepath}"
    print('Uploading {} => {}'.format(src_path, dst_path))
    # print(os.path.exists(src_path))
    bucket = get_bucket(bucket_namespace, bucket_name)
    # print('searching blob ...')
    blob = bucket.blob(dst_path)
    # print('start uploading ...')
    blob.upload_from_filename(src_path)
    print('finish uploading.')





