import os


class DatasetCatalog:
    DATA_DIR = "/tmp/kitti_three_2000_eval"
    DATASETS = {
        "kitti_train": {"root": "training/"},
        "kitti_test": {"root": "training/"},
    }

    @staticmethod
    def get(name):
        if "kitti" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = {"root": os.path.join(DatasetCatalog.DATA_DIR, attrs["root"])}
            return {"factory": "KITTIDataset", "args": args}
        raise RuntimeError(f"Dataset not available: {name}")


class ModelCatalog:
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)
        raise RuntimeError(f"Model not available: {name}")

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        return ModelCatalog.IMAGENET_MODELS[name]
