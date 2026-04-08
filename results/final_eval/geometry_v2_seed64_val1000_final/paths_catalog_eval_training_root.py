import os


class DatasetCatalog():
    DATA_DIR = r"/home/dy-jang/projects/v3/kitti_smoke_1280x384_lb"
    DATASETS = {
        "kitti_train": {"root": "training/"},
        "kitti_test": {"root": "training/"},
    }

    @staticmethod
    def get(name):
        if "kitti" in name:
            attrs = DatasetCatalog.DATASETS[name]
            return dict(
                factory="KITTIDataset",
                args=dict(root=os.path.join(DatasetCatalog.DATA_DIR, attrs["root"])),
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        return ModelCatalog.IMAGENET_MODELS[name]
