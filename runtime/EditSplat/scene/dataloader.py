from torch.utils.data import Dataset, DataLoader
from utils.camera_utils import cameraList_from_camInfos
from scene import Scene

class CameraDataset(Dataset):
    def __init__(self, scene):
        self.camera_list = scene.getTrainCameras().copy()
    
    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        camera = self.camera_list[idx]

        gt_image = camera.gt_image
        if hasattr(gt_image, "detach"):
            gt_image = gt_image.detach()
        if getattr(gt_image, "is_cuda", False):
            gt_image = gt_image.to(device="cpu", non_blocking=False)
        gt_image = gt_image.contiguous()

        return {
            'idx': idx, # data index
            'gt_image': gt_image,  # keep dataloader collation on CPU for stability
        }
