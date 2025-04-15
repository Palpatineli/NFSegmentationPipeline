from shutil import move
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from lib.infers.inferer_multi_stage_pipeline import InfererMultiStagePipeline
from lib.configs.config_3d_anisotropic_anatomic_unet import Config3DAnisotropicAnatomicUnet
from lib.configs.config_mrsegmentator import ConfigMRSegmentator
from lib.configs.config_probability_thresholding import ConfigProbabilityThresholding


model_dir = '/home/palpatine/alpaca/NFSegmentationPipeline/nf_segmentation_app/model'
model_conf = {'models': 'all', 'batch_size': '2', 'resample_only_in_2d': 'True'}

configs = {
    'config_3d_anisotropic_anatomic_unet': Config3DAnisotropicAnatomicUnet(),
    'config_mrsegmentator': ConfigMRSegmentator(),
    'config_probability_thresholding_medium': ConfigProbabilityThresholding()
}

components = {}

for n, task_config in configs.items():
    task_config.init(n, model_dir, model_conf, None, resample_only_in_2d=True)
    c = task_config.infer()
    c = c if isinstance(c, dict) else {n: c}
    for k, v in c.items():
        components[k] = v

task = InfererMultiStagePipeline(
    task_anatomy_segmentation=components["config_mrsegmentator"],
    task_neurofibroma_segmentation=components["config_3d_anisotropic_anatomic_unet"],
    task_thresholding=components["config_probability_thresholding_medium"],
    description="Multi-Stage Pipeline for neurofibroma segmentation",    
)

def main(image_path: Path):
    print(f'[Infer] {image_path.name}')
    res = task({'image': str(image_path)})
    move(res[0]['final'], str(image_path.parent.joinpath(image_path.stem + '_label.nii.gz')))

def _exists(path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    else:
        raise ArgumentTypeError(f'{path} does not exist')

if __name__ == "__main__":
    # image_path = Path('/home/palpatine/c/Downloads/sample/studies_20.nii')
    parser = ArgumentParser()
    parser.add_argument('image', type=_exists)
    args = parser.parse_args()
    image_path: Path = args.image
    if image_path.is_dir():
        for img in image_path.iterdir():
            if img.name.endswith('.nii') or img.name.endswith('.nii.gz'):
                if 'label' not in img.stem.lower():
                    main(img)
    else:
        main(image_path)
