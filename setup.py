import os
import shutil
import zipfile
import tarfile
import subprocess


def extract_zip(in_file, out_dir):
    zf = zipfile.ZipFile(in_file, 'r')
    zf.extractall(out_dir)
    zf.close()

def extract_tar(in_file, out_dir):
    tf = tarfile.open(in_file)
    tf.extractall(out_dir)
    tf.close()

def copy(in_file, out_dir):        
    cmd = ['cp', in_file, out_dir]
    subprocess.call(cmd)
    return 1
    
def _setup(data_dir, name='train'):
    print (f'Setting up {name}.')
    
    dataset = os.path.basename(os.path.normpath(data_dir))
    if os.path.isfile(os.path.join(data_dir, f'{name}.zip')):
        data_file = os.path.join(data_dir, f'{name}.zip')
        zip_file = True
    elif os.path.isfile(os.path.join(data_dir, f'{name}.tar')):
        data_file = os.path.join(data_dir, f'{name}.tar')
        zip_file = False
    else:
        raise ValueError('Invalid dataset path!')
    tmp_dir = os.path.join(os.environ['TMPDIR'], dataset)
    
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    copy(data_file, tmp_dir)
    if zip_file:
        if name in ['train']:
            extract_zip(os.path.join(tmp_dir, f'{name}.zip'), os.path.join(tmp_dir, name))
        else:
            extract_zip(os.path.join(tmp_dir, f'{name}.zip'), tmp_dir)
        os.remove(os.path.join(tmp_dir, f'{name}.zip'))
    else:
        extract_tar(os.path.join(tmp_dir, f'{name}.tar'), tmp_dir)
        os.remove(os.path.join(tmp_dir, f'{name}.tar'))

def setup_train(data_dir):
    _setup(data_dir, 'train')
    
def setup_val(data_dir):
    _setup(data_dir, 'val')


_CORRUPTIONS = ['noise', 'blur', 'weather', 'digital']

def setup_corruptions(data_dir, corruptions=_CORRUPTIONS):
    for corr in corruptions:
        _setup(data_dir, corr)
         
def setup_all(data_dir, cc_dir):
    setup_train(data_dir)
    setup_val(data_dir)
    setup_corruptions(cc_dir)


if __name__ == '__main__':
    setup_all()