import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import pycolmap


def get_cam_ray_dirs(camera):
    x = np.arange(camera.width, dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate([ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1)
    ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


class SegmentedCOLMAPDataset:
    def __init__(self, datadir, split, downsample):
        assert downsample in [1, 2, 4, 8]

        self.root_dir = datadir
        self.colmap_dir = os.path.join(datadir, "sparse/0/")
        self.split = split
        self.downsample = downsample

        if downsample == 1:
            
            images_dir = os.path.join(datadir, "object_mask")
        else:
            images_dir = os.path.join(datadir, f"object_mask_{downsample}")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory {images_dir} not found")

        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)

        if len(self.reconstruction.cameras) > 1:
            raise ValueError("Multiple cameras are not supported")

        names = sorted(im.name for im in self.reconstruction.images.values())
        indices = np.arange(len(names))

        if split == "train":
            names = list(np.array(names)[indices % 8 != 0])
        elif split == "test":
            names = list(np.array(names)[indices % 8 == 0])
        else:
            raise ValueError(f"Invalid split: {split}")

        names = list(str(name) for name in names)

        self.image_names = names

        # Get actual files in the masks directory (may have different names)
        actual_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])

        # Map COLMAP names to actual files by index
        all_colmap_names = sorted(im.name for im in self.reconstruction.images.values())
        self.name_to_file = {}
        for i, colmap_name in enumerate(all_colmap_names):
            if i < len(actual_files):
                self.name_to_file[colmap_name] = actual_files[i]

        self.image_paths = [os.path.join(images_dir, self.name_to_file[n]) for n in names]

        # Use the first actual file to determine resolution
        sample_png = self.image_paths[0]

        im = Image.open(sample_png)
        self.img_wh = im.size
        im.close()

        self.camera = list(self.reconstruction.cameras.values())[0]
        self.camera.rescale(self.img_wh[0], self.img_wh[1])

        self.fx = self.camera.focal_length_x
        self.fy = self.camera.focal_length_y

        cam_ray_dirs = get_cam_ray_dirs(self.camera)

        self.images = []
        for name in names:
            if name.startswith("test_"):
                continue
            image = None
            for image_id in self.reconstruction.images:
                image = self.reconstruction.images[image_id]
                if image.name == name:
                    break

            if image is None:
                raise ValueError(f"Image {name} not found in COLMAP reconstruction")

            self.images.append(image)

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        for image in tqdm(self.images):
            c2w = torch.tensor(image.cam_from_world.inverse().matrix(), dtype=torch.float32)
            self.poses.append(c2w)

            world_ray_dirs = torch.einsum("ij,kj->ik", cam_ray_dirs, c2w[:, :3])
            world_ray_origins = c2w[:, 3] + torch.zeros_like(cam_ray_dirs)
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)

            # Use mapped filename
            im_path = os.path.join(images_dir, self.name_to_file[image.name])

            im = Image.open(im_path).convert("L")
            rgbs = torch.tensor(np.array(im), dtype=torch.float32)
            rgbs.unsqueeze_(2)
            im.close()
            

            self.all_rays.append(world_rays)
            self.all_rgbs.append(rgbs)

        self.poses = torch.stack(self.poses)
        self.all_rays = torch.stack(self.all_rays)
        self.all_rgbs = torch.stack(self.all_rgbs)

        self.points3D = []
        self.points3D_color = []
        for point in self.reconstruction.points3D.values():
            self.points3D.append(point.xyz)
            self.points3D_color.append(point.color)

        self.points3D = torch.tensor(np.array(self.points3D), dtype=torch.float32)
        self.points3D_color = torch.tensor(np.array(self.points3D_color), dtype=torch.float32)
        self.points3D_color = self.points3D_color / 255.0