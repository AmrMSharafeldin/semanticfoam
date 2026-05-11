import os
import uuid
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
import tqdm
import radfoam
from radfoam_model.render import TraceRays
from radfoam_model.utils import *
import fpsample
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class RadFoamScene(torch.nn.Module):

    def __init__(
        self,
        args,
        points=None,
        points_colors=None,
        cameras=None,
        device=torch.device("cuda"),
        random_segments_init = True,
        attr_dtype=torch.float32,
    ):
        super().__init__()

        self.device = device
        self.attr_dtype = attr_dtype
        if cameras is not None:
            self.cameras = cameras.to(device)
        else:
            self.cameras = None
        self.sh_degree = args.sh_degree
        self.num_init_points = args.init_points
        self.num_final_points = args.final_points
        self.activation_scale = args.activation_scale
        self.identity_dim = args.identity_dim
        self.unfreeze_seg_iter = args.unfreeze_seg_iter

        self.render_mode = "rgb" # "rgb" or "segmentation"
        

        if points is not None :
            self.initialize_from_pcd(points, points_colors)
        else:
            self.random_initialize()

        if not random_segments_init:
            print(
                "Initializing identity encoding with random segments"
            )
            self.initialize_identity_encoding(
                num_clusters=args.num_identity_clusters,
                identity_dim=args.identity_dim,
            )
        self.att_dc = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3,
                device=self.device,
                dtype=self.attr_dtype,
            )
        )
        self.att_sh = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3 * ((1 + self.sh_degree) * (1 + self.sh_degree) - 1),
                device=device,
                dtype=self.attr_dtype,
            )
        )

        print(self.identity_dim)
        self.identity_encoding = nn.Parameter(
            torch.empty(
                self.num_init_points,
                self.identity_dim,
                device=device,
                dtype=self.attr_dtype,
                requires_grad=True
            )
    )
        torch.nn.init.xavier_uniform_(self.identity_encoding.data)
        print(self.identity_encoding.shape)





        self.pipeline = radfoam.create_pipeline(self.sh_degree,  self.attr_dtype , self.identity_dim) ## TODO: change the pipeline create  in cuda [X]

    def random_initialize(self):
        primal_points = (
            torch.randn(self.num_init_points, 3, device=self.device) * 25
        )
        self.triangulation = radfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        self.att_dc = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3,
                device=self.device,
                dtype=self.attr_dtype,
            )
        )

        density = torch.zeros(
            self.num_init_points, 1, device=self.device, dtype=self.attr_dtype
        )
        self.density = nn.Parameter(density[perm])

    def initialize_from_pcd(self, points, points_colors):
        print(f"Initializing from PCD with {points.shape[0]} points")
        points = points.to(self.device)
        points_mean = points.mean(dim=0, keepdim=True)
        points_std = points.std(dim=0, keepdim=True)
        points_colors = points_colors.to(self.device)

        num_random = 5_000
        random = (
            torch.randn([num_random, 3], device=self.device) * points_std
            + points_mean
        )

        num_samples = int(0.8* points.shape[0])
        print(
            f"Starting with {num_samples} points from {points.shape[0]} COLMAP points"
        )
        points_idx = fpsample.bucket_fps_kdtree_sampling(
            points.cpu().numpy(), num_samples
        )
        points_idx = torch.tensor(
            points_idx, device=self.device, dtype=torch.long
        )
        samp_points = points[points_idx]
        samp_colors = points_colors[points_idx]

        primal_points = torch.cat([samp_points, random], dim=0)
        primal_density = torch.cat(
            [
                torch.rand(samp_colors.shape[0], 1, dtype=self.attr_dtype),
                -0.5 * torch.ones(num_random, 1, dtype=self.attr_dtype),
            ],
            dim=0,
        ).to(self.device)

        torch.cuda.empty_cache()

        self.triangulation = radfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        self.density = nn.Parameter(primal_density)
        self.num_init_points = self.primal_points.shape[0]

        self.update_triangulation(rebuild=False)

        self.density = nn.Parameter(primal_density)

        self.num_init_points = self.primal_points.shape[0]
    
    def initialize_identity_encoding(self, num_clusters=16, identity_dim=16):
        """
        Assign initial identity encodings to Gaussians based on spatial proximity.
        """
        from sklearn.cluster import KMeans

        # Detach and move to CPU for clustering
        with torch.no_grad():
            positions = self.primal_points.detach().cpu().numpy()

        # Perform K-means clustering in 3D space
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(positions)
        labels = kmeans.labels_  # shape: [num_points]

        # Create a fixed embedding table for each class (either random or learnable base vectors)
        identity_table = torch.randn(num_clusters, identity_dim)

        # Assign identity vectors to each point
        identity_vectors = identity_table[torch.tensor(labels)]  # shape: [num_points, identity_dim]

        # Register as learnable parameter
        self.identity_encoding = nn.Parameter(identity_vectors.to(self.device, dtype=self.attr_dtype))


    def permute_points(self, permutation):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "env" not in group["name"]:
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][
                        permutation
                    ]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][
                        permutation
                    ]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][permutation].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][permutation].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        self.identity_encoding = optimizable_tensors["identity_encoding"]


    def update_triangulation(self, rebuild=True, incremental=False):
        if not self.primal_points.isfinite().all():
            raise RuntimeError("NaN in points")

        needs_permute = False
        perturbation = 1e-6
        del_points = self.primal_points
        failures = 0
        while rebuild:
            if failures > 25:
                raise RuntimeError("aborted triangulation after 25 attempts")
            try:
                needs_permute = self.triangulation.rebuild(
                    del_points, incremental=incremental
                )
                break
            except radfoam.TriangulationFailedError as e:
                print("caught: ", e)
                perturbation *= 2
                failures += 1
                incremental = False
                with torch.no_grad():
                    del_points = (
                        self.primal_points
                        + perturbation * torch.randn_like(self.primal_points)
                    )

        if failures > 5:
            with torch.no_grad():
                self.primal_points.copy_(del_points)

        if needs_permute:
            perm = self.triangulation.permutation().to(torch.long)
            self.permute_points(perm)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)


        self.point_adjacency = self.triangulation.point_adjacency()
        self.point_adjacency_offsets = (
            self.triangulation.point_adjacency_offsets()
        )

        self.tets = self.triangulation.tets()
        self.tet_adjacency = self.triangulation.tet_adjacency()
        self.faces = self.triangulation.faces()
        self.edges = self.triangulation.edges()

   

    def get_primal_density(self):
        return self.activation_scale * F.softplus(self.density, beta=10)



    def get_primal_attributes(self):
        C0 = 0.28209479177387814

        if self.render_mode == "rgb" or not hasattr(self, "segmentation_index"):
            return torch.cat([self.att_dc, self.att_sh], dim=-1)

        elif self.render_mode == "segmentation":
            cls = self.segmentation_index.long()
            palette = tab20_palette_torch(num_classes=256, device=cls.device)  # tab20 has 20 entries

            class_ids = (cls % palette.shape[0]).clamp(0, palette.shape[0]-1)
            rgb_map = palette[class_ids]

            att_dc_cls = (rgb_map - 0.5) / C0
            att_sh_cls = torch.zeros_like(self.att_sh)

            return torch.cat([att_dc_cls, att_sh_cls], dim=-1)


    def get_trace_data(self):
        points = self.primal_points
        attributes = torch.cat(
            [self.get_primal_attributes(), self.get_primal_density() , self.identity_encoding],
            dim=-1,
        ).to(self.attr_dtype)


        
        point_adjacency = self.point_adjacency
        point_adjacency_offsets = self.point_adjacency_offsets

        return points, attributes, point_adjacency, point_adjacency_offsets

    def show(self, loop_fn=lambda v: None, iterations=None, **viewer_kwargs):
        radfoam.run_with_viewer(
            self.pipeline, loop_fn, total_iterations=iterations, **viewer_kwargs
        )

    def get_starting_point(self, rays, points, aabb_tree):
        with torch.no_grad():
            camera_origins = rays[..., :3]
            unique_cameras, inverse_indices = torch.unique(
                camera_origins, dim=0, return_inverse=True
            )

            nn_inds = radfoam.nn(points, aabb_tree, unique_cameras).long()

            start_point = nn_inds[inverse_indices]
            return start_point.type(torch.uint32)


    # TODO change the forward pass in cuda 
    def forward(
        self,
        rays,
        start_point=None,
        depth_quantiles=None,
        return_contribution=False,
    ):
        points, attributes, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data()
        )

        if start_point is None:
            start_point = self.get_starting_point(rays, points, self.aabb_tree)
        else:
            start_point = torch.broadcast_to(start_point, rays.shape[:-1])
        return TraceRays.apply(
            self.pipeline,
            points,
            attributes,
            point_adjacency,
            point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles,
            return_contribution,
        )

    def total_variation_loss(self):
            edges = self.edges.long()
            identity = self.identity_encoding
            identity_diff = (identity[edges[:, 0]] - identity[edges[:, 1]]).abs()

            circumcenters = tetrahedron_circumcenters(
                self.primal_points, self.tets.long()
            ).detach()

            dface_area = radfoam.dface_area(
                self.tets,
                self.tet_adjacency,
                self.edges,
                circumcenters,
            )
            dface_area = dface_area.clamp(min=1)
            # print("Max face area for TV:", dface_area.max().item())
            # print("Min face area for TV:", dface_area.min().item())
            tv_loss = identity_diff * dface_area
            return tv_loss
    
    def update_viewer(self, viewer):
        points, attributes, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data()
        )
        viewer.update_scene(
            points,
            attributes,
            point_adjacency,
            point_adjacency_offsets,
            self.aabb_tree,
        )

    def declare_optimizer(self, args, warmup, max_iterations):
        params = [
            {
                "params": self.primal_points,
                "lr": args.points_lr_init,
                "name": "primal_points",
            },
            {
                "params": self.density,
                "lr": args.density_lr_init,
                "name": "density",
            },
            {
                "params": self.att_dc,
                "lr": args.attributes_lr_init,
                "name": "att_dc",
            },
            {
                "params": self.att_sh,
                "lr": args.attributes_lr_init,
                "name": "att_sh",
            },
            {
                "params": self.identity_encoding,
                "lr": args.attributes_lr_init,
                "name": "identity_encoding",
            },
        ]

        self.optimizer = torch.optim.Adam(params, eps=1e-15)
        self.xyz_scheduler_args = get_cosine_lr_func(
            lr_init=args.points_lr_init,
            lr_final=args.points_lr_final,
            max_steps=args.freeze_points,
        )
        self.den_scheduler_args = get_cosine_lr_func(
            lr_init=args.density_lr_init,
            lr_final=args.density_lr_final,
            warmup_steps=warmup,
            max_steps=max_iterations,
        )
        self.attr_dc_scheduler_args = get_cosine_lr_func(
            lr_init=args.attributes_lr_init,
            lr_final=args.attributes_lr_final,
            max_steps=max_iterations,
        )
        self.attr_rest_scheduler_args = get_cosine_lr_func(
            lr_init=args.sh_factor * args.attributes_lr_init,
            lr_final=args.sh_factor * args.attributes_lr_final,
            warmup_steps=max_iterations // 5,
            max_steps=max_iterations,
        )


        self.identity_scheduler_args = get_cosine_lr_func(
            lr_init=args.sh_factor * args.attributes_lr_init,
            lr_final=args.sh_factor * args.attributes_lr_final,
            warmup_steps=(max_iterations - self.unfreeze_seg_iter) // 5,
            max_steps=(max_iterations - self.unfreeze_seg_iter),
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density":
                lr = self.den_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_dc":
                lr = self.attr_dc_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_sh":
                lr = self.attr_rest_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "identity_encoding":
                if iteration >= self.unfreeze_seg_iter:
                    shifted_iter = iteration - self.unfreeze_seg_iter
                    lr = self.identity_scheduler_args(shifted_iter)
                    param_group["lr"] = lr
                else:
                    param_group["lr"] = 0.0

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, prune_mask):
        valid_points_mask = ~prune_mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        self.density = optimizable_tensors["density"]
        self.identity_encoding = optimizable_tensors["identity_encoding"]

    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in new_params.keys():
                assert len(group["params"]) == 1
                stored_tensor = group["params"][0]
                extension_tensor = new_params[group["name"]]
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_params):
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        self.density = optimizable_tensors["density"]
        self.identity_encoding = optimizable_tensors["identity_encoding"]

    def prune_and_densify(
        self, point_error, point_contribution, upsample_factor=1.2
    ):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            primal_error_accum = point_error.clip(min=0).squeeze()
            points, _, point_adjacency, point_adjacency_offsets = (
                self.get_trace_data()
            )
            farthest_neighbor, cell_radius = radfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            self_mask = point_contribution > 1e-2
            neighbor_mask = self_mask.long()[point_adjacency.long()]
            neighbor_mask = torch.cat(
                [neighbor_mask, torch.zeros_like(neighbor_mask[:1])], dim=0
            )
            nsum = torch.cumsum(neighbor_mask, dim=0)

            offsets = point_adjacency_offsets.long()
            n_masked_adj = nsum[offsets[1:]] - nsum[offsets[:-1]]

            contrib_mask = ((n_masked_adj == 0) & ~self_mask).squeeze()
            cell_size_mask = cell_radius < 1e-1
            prune_mask = contrib_mask * cell_size_mask

            primal_contribution_accum = point_contribution.squeeze()
            mask = primal_contribution_accum < 1e-3
            self.density[mask] = -1

            perturbation = 0.25 * (points[farthest_neighbor] - points)
            delta = torch.randn_like(perturbation)
            delta /= delta.norm(dim=-1, keepdim=True)
            perturbation += (
                0.1 * perturbation.norm(dim=-1, keepdim=True) * delta
            )

            num_sample_points = num_new_points
            sampled_inds = torch.multinomial(
                primal_error_accum * cell_radius,
                num_sample_points,
                replacement=False,
            )
            sampled_points = (points + perturbation)[sampled_inds]

            new_params = {
                "primal_points": sampled_points,
                "att_dc": self.att_dc[sampled_inds],
                "att_sh": self.att_sh[sampled_inds],
                "density": self.density[sampled_inds],
                "identity_encoding": self.identity_encoding[sampled_inds],
            }

            prune_mask = torch.cat(
                (
                    prune_mask,
                    torch.zeros(
                        sampled_points.shape[0],
                        device=prune_mask.device,
                        dtype=bool,
                    ),
                )
            )

            self.densification_postfix(new_params)
                # print("Shapes after densification:")
                # print("att_dc:", self.att_dc.shape)
                # print("att_sh:", self.att_sh.shape)
                # print("density:", self.density.shape)
                # print("identity_encoding:", self.identity_encoding.shape)

            self.prune_points(prune_mask)


    def collect_error_map(self, data_handler, white_bkg=True, downsample=2):
        rays, rgbs = data_handler.rays, data_handler.rgbs

        points, _, _, _ = self.get_trace_data()
        start_points = self.get_starting_point(
            rays[:, 0, 0].cuda(), points, self.aabb_tree
        )

        ray_batch_fetcher = radfoam.BatchFetcher(
            rays, batch_size=1, shuffle=False
        )
        rgb_batch_fetcher = radfoam.BatchFetcher(
            rgbs, batch_size=1, shuffle=False
        )

        point_error_accum = torch.zeros_like(self.primal_points[..., 0:1])
        point_contribution_accum = torch.zeros_like(
            self.primal_points[..., 0:1]
        )
        rgb_loss = nn.L1Loss(reduction="none")

        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()

            d = torch.randint(0, downsample, (2,))
            ray_batch = ray_batch[:, d[0] :: downsample, d[1] :: downsample, :]
            rgb_batch = rgb_batch[:, d[0] :: downsample, d[1] :: downsample, :]

            rgba_output, _ , _, contribution, _, errbox = self.forward(
                ray_batch, start_points[i], return_contribution=True
            )
            opacity = rgba_output[..., -1:]
            if white_bkg:
                rgb_output = rgba_output[..., :3] + (1 - opacity)
            else:
                rgb_output = rgba_output[..., :3]

            color_loss = rgb_loss(rgb_batch, rgb_output).mean(dim=-1)

            color_loss.sum().backward()
            point_error_accum += self.primal_points.grad.norm(
                dim=-1, keepdim=True
            ).detach()
            point_contribution_accum = torch.maximum(
                point_contribution_accum, contribution.detach()
            )
            torch.cuda.synchronize()

            self.optimizer.zero_grad(set_to_none=True)

        return point_error_accum, point_contribution_accum

    def save_ply(self, ply_path):
        points = self.primal_points.detach().float().cpu().numpy()
        density = self.get_primal_density().detach().float().cpu().numpy()
        color_attributes = (
            self.get_primal_attributes().detach().float().cpu().numpy()
        )
        adjacency = self.point_adjacency.cpu().numpy()
        adjacency_offsets = self.point_adjacency_offsets.cpu().numpy()

        C0 = 0.28209479177387814
        r = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 0]), 0, 255),
            dtype=np.uint8,
        )
        g = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 1]), 0, 255),
            dtype=np.uint8,
        )
        b = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 2]), 0, 255),
            dtype=np.uint8,
        )

        vertex_data = []
        for i in tqdm.trange(points.shape[0]):
            vertex_data.append(
                (
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    r[i],
                    g[i],
                    b[i],
                    density[i, 0],
                    adjacency_offsets[i + 1],
                    *[
                        color_attributes[i, 3 + j]
                        for j in range(color_attributes.shape[1] - 3)
                    ],
                )
            )

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
            ("density", np.float32),
            ("adjacency_offset", np.uint32),
        ]

        for i in range(self.att_sh.shape[1]):
            dtype.append(("color_sh_{}".format(i), np.float32))

        vertex_data = np.array(vertex_data, dtype=dtype)
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        adjacency_data = np.array(adjacency, dtype=[("adjacency", np.uint32)])
        adjacency_element = PlyElement.describe(adjacency_data, "adjacency")

        PlyData([vertex_element, adjacency_element]).write(ply_path)

    def save_pt(self, pt_path):
        points = self.primal_points.detach().float().cpu()
        density = self.density.detach().float().cpu()
        color_dc = self.att_dc.detach().float().cpu()
        color_sh = self.att_sh.detach().float().cpu()
        adjacency = self.point_adjacency.cpu()
        adjacency_offsets = self.point_adjacency_offsets.cpu()
        identity = self.identity_encoding.detach().float().cpu()   # 🔹 NEW

        scene_data = {
            "xyz": points,
            "density": density,
            "color_dc": color_dc,
            "color_sh": color_sh,
            "adjacency": adjacency.long(),
            "adjacency_offsets": adjacency_offsets.long(),
            "identity_encoding": identity,   # 🔹 NEW
        }
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        scene_data = torch.load(pt_path, map_location=self.device)

        self.primal_points = nn.Parameter(scene_data["xyz"].to(self.device))
        self.density = nn.Parameter(scene_data["density"].to(self.device))
        self.att_dc = nn.Parameter(
            scene_data["color_dc"].to(self.attr_dtype).to(self.device)
        )

        exp_sh_coeffs = 3 * ((1 + self.sh_degree) * (1 + self.sh_degree) - 1)
        got_sh_coeffs = scene_data["color_sh"].shape[-1]
        assert (
            exp_sh_coeffs == got_sh_coeffs
        ), f"Expected {exp_sh_coeffs} SH coeffs per-point, got {got_sh_coeffs}"
        self.att_sh = nn.Parameter(
            scene_data["color_sh"].to(self.attr_dtype).to(self.device)
        )

        self.point_adjacency = scene_data["adjacency"].to(self.device).to(torch.uint32)
        self.point_adjacency_offsets = scene_data["adjacency_offsets"].to(self.device).to(torch.uint32)

        # 🔹 NEW: Restore identity encoding if present
        if "identity_encoding" in scene_data:
            self.identity_encoding = nn.Parameter(
                scene_data["identity_encoding"].to(self.attr_dtype).to(self.device)
            )
            print(f"Loaded identity_encoding with shape {self.identity_encoding.shape}")
        else:
            print(" No identity_encoding found in checkpoint, initializing randomly.")
            self.identity_encoding = nn.Parameter(
                torch.empty(
                    self.primal_points.shape[0],
                    self.identity_dim,
                    device=self.device,
                    dtype=self.attr_dtype,
                    requires_grad=True,
                )
            )
            torch.nn.init.xavier_uniform_(self.identity_encoding.data)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)
        
        self.segmentation_index = torch.zeros(
        self.primal_points.shape[0],
        dtype=torch.long,
        device=self.device)


    def freeze_identity_encoding(self):
        """Stop gradients for identity_encoding (keep everything else trainable)."""
        self.identity_encoding.requires_grad_(False)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "identity_encoding":
                param_group["params"][0].requires_grad = False

    def freeze_all_but_identity_encoding(self):
        """Freeze all parameters except identity_encoding."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "identity_encoding":
                param_group["params"][0].requires_grad = True
                self.identity_encoding.requires_grad_(True)
            else:
                param_group["params"][0].requires_grad = False
    
    
    def shutdown_density(self, refined_mask):
        self.density[refined_mask] = 1e-9
        print(f"[INFO] Density shutdown applied to {refined_mask.sum().item()} points.")


    def remove_points(self, refined_mask , density_thresh=.5):

        primal_density = self.get_primal_density().view(-1)
        high_density_mask = (primal_density > density_thresh).to(refined_mask.device)
        
        neighbor_matrix = build_neighbor_matrix(self)
        masked_neighbors = neighbor_matrix[refined_mask]
        masked_neighbors = masked_neighbors[masked_neighbors != -1]
        neighbor_ids = torch.unique(masked_neighbors)

        # neighbors outside the mask
        halo_ids = neighbor_ids[~refined_mask[neighbor_ids]]


        # expanded mask = mask ∪ halo
        expanded_mask = refined_mask.clone()
        expanded_mask = expanded_mask & high_density_mask
        expanded_mask[halo_ids] = True
        self.prune_points(expanded_mask)
        self.update_triangulation(rebuild=True)
        print(f"[INFO] Removed {refined_mask.sum().item()} points from triangulation.")


    def duplicate_points(
        self,
        refined_mask,
        translation=torch.tensor([1.0, 0.0, 0.0], device="cuda"),
        ghost_density=1e-9,
        atol=1e-2,
    ):
        """
        Expand mask by 1-ring neighbors, duplicate them with translation.
        Original mask -> normal density, neighbor halo -> ghost density.
        """
        # === Step 1: Expand mask by 1-ring neighbors ===
        neighbor_matrix = build_neighbor_matrix(self)
        masked_neighbors = neighbor_matrix[refined_mask]
        masked_neighbors = masked_neighbors[masked_neighbors != -1]
        neighbor_ids = torch.unique(masked_neighbors)

        # neighbors outside the mask
        halo_ids = neighbor_ids[~refined_mask[neighbor_ids]]

        # expanded mask = mask ∪ halo
        expanded_mask = refined_mask.clone()
        expanded_mask[halo_ids] = True

        # === Step 2: Translate duplicated set ===
        translated_positions = self.primal_points[expanded_mask] + translation

        dup_dc       = self.att_dc[expanded_mask].clone()
        dup_sh       = self.att_sh[expanded_mask].clone()
        dup_density  = self.density[expanded_mask].clone()
        dup_identity = self.identity_encoding[expanded_mask].clone()

        # set densities of halo part to ghost
        expanded_ids = torch.nonzero(expanded_mask, as_tuple=True)[0]
        halo_mask_in_expanded = torch.isin(expanded_ids, halo_ids)
        dup_density[halo_mask_in_expanded] = ghost_density

        # === Step 3: Insert new points ===
        new_params = {
            "primal_points": translated_positions,
            "att_dc": dup_dc,
            "att_sh": dup_sh,
            "density": dup_density,
            "identity_encoding": dup_identity,
        }
        self.densification_postfix(new_params)
        self.update_triangulation(rebuild=True)

        print(f"[INFO] Duplicated {expanded_mask.sum().item()} points (mask + 1-ring halo).")
        print(f"       Core mask kept normal, halo set to ghost density.")





    def move_points(self, refined_mask, translation=torch.tensor([1.0,0.0,0.0])):
        translated_positions = self.primal_points[refined_mask] + translation
        match_mask = torch.zeros(self.primal_points.shape[0], dtype=torch.bool, device=self.device)
        for tp in translated_positions:
            match_mask |= torch.isclose(self.primal_points, tp, rtol=0, atol=1e-2).all(dim=1)
        match_mask &= ~refined_mask
        if match_mask.any():
            self.prune_points(match_mask)
            refined_mask = refined_mask[~match_mask]
            print(f"[INFO] Pruned {match_mask.sum().item()} points at destination.")

        self.primal_points[refined_mask] = translated_positions
        self.update_triangulation(rebuild=True, incremental=True)
        print(f"[INFO] Moved {refined_mask.sum().item()} points by {translation.tolist()}.")

    def save_asset(self, refined_mask, save_path_base="asset"):
        if not refined_mask.any():
            print("[DEBUG] No points selected to save.")
            return

        # Build subset scene
        subset_scene = RadFoamScene.__new__(RadFoamScene)
        subset_scene.device = self.device
        subset_scene.attr_dtype = self.attr_dtype
        subset_scene.sh_degree = self.sh_degree
        subset_scene.identity_dim = self.identity_dim
        subset_scene.activation_scale = self.activation_scale
        subset_scene.render_mode = self.render_mode

        subset_scene.primal_points = self.primal_points[refined_mask].detach().clone()
        subset_scene.density = self.density[refined_mask].detach().clone()
        subset_scene.att_dc = self.att_dc[refined_mask].detach().clone()
        subset_scene.att_sh = self.att_sh[refined_mask].detach().clone()
        subset_scene.identity_encoding = self.identity_encoding[refined_mask].detach().clone()

        subset_scene.point_adjacency = torch.zeros((0,), dtype=torch.int32)
        subset_scene.point_adjacency_offsets = torch.zeros(
            (subset_scene.primal_points.shape[0]+1,), dtype=torch.int32)

        # Save both pt + ply
        pt_path = f"{save_path_base}.pt"
        ply_path = f"{save_path_base}.ply"
        subset_scene.save_pt(pt_path)
        subset_scene.save_ply(ply_path)

        print(f"[INFO] Saved asset with {subset_scene.primal_points.shape[0]} points → {pt_path}, {ply_path}")\


    def import_asset(self, loaded_scene,
                    translation=[0, 0, 0],
                    scale_factor=1.0,
                    rotation_degrees=[0, 0, 0],
                    logger=None,
                    debug=False):

        log = logger.info if logger is not None else print

        # ---------------------------------------------------------------
        # 0. Remove previous object (id = 79)
        # ---------------------------------------------------------------
        # target_mask = self.segmentation_index == 117
        # self.density[target_mask] = -1e9

        # ---------------------------------------------------------------
        # 1. Extract asset tensors
        # ---------------------------------------------------------------
        pts  = loaded_scene.primal_points.clone().to(self.device)
        dens = loaded_scene.density.clone().to(self.device)
        dc   = loaded_scene.att_dc.clone().to(self.device)
        sh   = loaded_scene.att_sh.clone().to(self.device)
        ide  = loaded_scene.identity_encoding.clone().to(self.device)

        # ---------------------------------------------------------------
        # 1B. Debug subsampling: 10% but >= 5000 points
        # ---------------------------------------------------------------
        if debug:
            total = pts.shape[0]
            target = min(max(5000, total // 10), total)

            log(f"[DEBUG] Random subsampling asset: {target} / {total} points")

            idx = torch.randperm(total, device=self.device)[:target]
            pts, dens, dc, sh, ide = pts[idx], dens[idx], dc[idx], sh[idx], ide[idx]

        new_points   = pts
        new_density  = dens
        new_dc       = dc
        new_sh       = sh
        new_id       = ide

        # ---------------------------------------------------------------
        # 2. Center object in local coords
        # ---------------------------------------------------------------
        asset_center = new_points.mean(dim=0, keepdim=True)
        new_points = new_points - asset_center

        # ---------------------------------------------------------------
        # 2B. OUTLIER REMOVAL (MAD-based)
        # ---------------------------------------------------------------
        dist = torch.norm(new_points, dim=1)

        median = torch.median(dist)
        mad = torch.median(torch.abs(dist - median)) + 1e-6
        k = 6.0
        threshold = median + k * mad

        keep_mask = dist < threshold
        num_keep = keep_mask.sum().item()
        num_total = new_points.shape[0]

        log(f"[CLEANUP] Removing outliers: keeping {num_keep}/{num_total} points "
            f"(threshold={threshold.item():.3f})")

        new_points = new_points[keep_mask]
        new_density = new_density[keep_mask]
        new_dc = new_dc[keep_mask]
        new_sh = new_sh[keep_mask]
        new_id = new_id[keep_mask]

        # ---------------------------------------------------------------
        # 3. Apply scale
        # ---------------------------------------------------------------
        new_points = new_points * scale_factor

        # ---------------------------------------------------------------
        # 4. Apply rotation around object center
        # ---------------------------------------------------------------
        if rotation_degrees is not None:
            if isinstance(rotation_degrees, (int, float)):
                rx, ry, rz = 0.0, float(rotation_degrees), 0.0
            else:
                rx, ry, rz = rotation_degrees

            rx = torch.deg2rad(torch.tensor(rx, device=self.device))
            ry = torch.deg2rad(torch.tensor(ry, device=self.device))
            rz = torch.deg2rad(torch.tensor(rz, device=self.device))

            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(rx), -torch.sin(rx)],
                [0, torch.sin(rx),  torch.cos(rx)]
            ], device=self.device)

            Ry = torch.tensor([
                [ torch.cos(ry), 0, torch.sin(ry)],
                [0,              1, 0],
                [-torch.sin(ry), 0, torch.cos(ry)]
            ], device=self.device)

            Rz = torch.tensor([
                [torch.cos(rz), -torch.sin(rz), 0],
                [torch.sin(rz),  torch.cos(rz), 0],
                [0, 0, 1]
            ], device=self.device)

            R = Rz @ Ry @ Rx
            new_points = new_points @ R.T

        # ---------------------------------------------------------------
        # 5. Move to target location in the scene
        # ---------------------------------------------------------------
        scene_center = self.primal_points.mean(dim=0, keepdim=True)
        #target_center = self.primal_points[target_mask].mean(dim=0, keepdim=True)

        new_points = new_points + scene_center

        # ---------------------------------------------------------------
        # 6. User translation
        # ---------------------------------------------------------------
        new_points = new_points + torch.tensor(translation, device=self.device)

        log(f"[RADFOAM] Inserting asset at asset center: "
            f"{asset_center.view(-1).cpu().detach().numpy()}")

        move_vec = scene_center + torch.tensor(translation, device=self.device)
        log(f"[RADFOAM] total translation vector : "
            f"{move_vec.view(-1).cpu().detach().numpy()}")

        # ---------------------------------------------------------------
        # 7. Merge into RadFoam scene
        # ---------------------------------------------------------------
        new_params = {
            "primal_points": new_points,
            "density": new_density,
            "att_dc": new_dc,
            "att_sh": new_sh,
            "identity_encoding": new_id,
        }

        self.densification_postfix(new_params)
        self.update_triangulation(rebuild=True)

        log("[RADFOAM] Asset inserted (with outlier cleanup, rotation, scale, debug).")





    







    def update_segmentation_indices(self, classifier, classifier_args):
        """Run classifier on per-point features, store argmax class index per point, and save them in ./dummy."""
        with torch.no_grad():
            _, attributes, _, _ = self.get_trace_data()
            seg_features = attributes[..., -classifier_args.input_dim:]

            logits_pts = classifier(seg_features)
            probs = torch.softmax(logits_pts, dim=-1)
            self.segmentation_index = probs.argmax(dim=-1).to(self.device)

            # Ensure save directory exists
            save_dir = "./dummy"
            os.makedirs(save_dir, exist_ok=True)

            # Save tensor and NumPy versions
            torch.save(self.segmentation_index.cpu(), os.path.join(save_dir, "segmentation_indices.pt"))
            np.save(os.path.join(save_dir, "segmentation_indices.npy"), self.segmentation_index.cpu().numpy())

            print(f"[INFO] Saved segmentation indices to {save_dir}")

    

    def set_render_mode(self, mode: str):
        assert mode in ["rgb", "segmentation"], "Invalid render mode"
        self.render_mode = mode

    def get_render_mode(self):
        return self.render_mode


    def save_scene_and_asset_pointclouds(
            self,
            save_dir="./export_pc",
            refined_mask=None,
            density_thresh=0.5,
            max_num_obj=256,
            object_name=None   # optional override
        ):
        """
        Exports:

            scene.pts.input.ply
            scene.pts.instance_pred.ply
            <object>.pts.input.ply
            <object>.pts.instance_pred.ply

        Uses your EXACT id2rgb palette.
        Filters out points with primal_density < density_thresh.
        """

        os.makedirs(save_dir, exist_ok=True)
        print("[SAVE] Preparing scene data...")

        # ----------------------------------------
        # LOAD
        # ----------------------------------------
        pts = self.primal_points.detach().cpu()
        att_dc = self.att_dc.detach().cpu()
        seg = self.segmentation_index.detach().cpu().long()
        primal_density = self.get_primal_density().detach().cpu().squeeze()

        # ----------------------------------------
        # FILTER BY DENSITY
        # ----------------------------------------
        keep_mask = primal_density >= density_thresh
        print(f"[FILTER] Keeping {keep_mask.sum().item()}/{pts.shape[0]} points")

        pts = pts[keep_mask]
        att_dc = att_dc[keep_mask]
        seg = seg[keep_mask]

        if refined_mask is not None:
            refined_mask = refined_mask.cpu().bool()
            refined_mask = refined_mask[keep_mask]

        # ----------------------------------------
        # COLORIZATION
        # ----------------------------------------

        # SH-DC → RGB
        C0 = 0.28209479177387814
        rgb = 255 * torch.clamp(0.5 + C0 * att_dc[:, :3], 0, 1)
        rgb = rgb.to(torch.uint8)

        # segmentation → id2rgb
        seg_np = seg.numpy()
        seg_colors = np.zeros((seg_np.shape[0], 3), dtype=np.uint8)

        for cls_id in np.unique(seg_np):
            seg_colors[seg_np == cls_id] = id2rgb(int(cls_id), max_num_obj)

        seg_colors = torch.from_numpy(seg_colors)

        # ----------------------------------------
        # PLY WRITER
        # ----------------------------------------
        def write_ply(path, xyz, colors):
            xyz_np = xyz.numpy()
            colors_np = colors.numpy()

            arr = np.zeros(xyz_np.shape[0], dtype=[
                ("x","f4"),("y","f4"),("z","f4"),
                ("red","u1"),("green","u1"),("blue","u1")
            ])
            arr["x"] = xyz_np[:,0]
            arr["y"] = xyz_np[:,1]
            arr["z"] = xyz_np[:,2]
            arr["red"]   = colors_np[:,0]
            arr["green"] = colors_np[:,1]
            arr["blue"]  = colors_np[:,2]

            PlyData([PlyElement.describe(arr, "vertex")]).write(path)
            print(f"[SAVE] → {path}  ({len(arr)} points)")

        # ----------------------------------------
        # SCENE EXPORT
        # ----------------------------------------
        write_ply(
            f"{save_dir}/scene.pts.input.ply",
            pts,
            rgb
        )

        write_ply(
            f"{save_dir}/scene.pts.instance_pred.ply",
            pts,
            seg_colors
        )

        # ----------------------------------------
        # ASSET EXPORT
        # ----------------------------------------
        if refined_mask is None or refined_mask.sum().item() == 0:
            print("[INFO] No asset mask → skipping object files.")
            return

        # random name if user does not provide one
        if object_name is None:
            object_name = str(uuid.uuid4())[:8]   # short random ID

        print(f"[ASSET] Object name: {object_name}")

        pts_obj = pts[refined_mask]
        rgb_obj = rgb[refined_mask]
        seg_obj = seg_colors[refined_mask]

        write_ply(
            f"{save_dir}/{object_name}.pts.input.ply",
            pts_obj,
            rgb_obj
        )

        write_ply(
            f"{save_dir}/{object_name}.pts.instance_pred.ply",
            pts_obj,
            seg_obj
        )

        print(f"[DONE] Saved scene + object pointclouds → {save_dir}")















def build_neighbor_matrix(scene, pad_value=-1):
    """
    Build a [N, max_degree] neighbor matrix from CSR adjacency.
    Each row: indices of neighbors of that point, padded with pad_value.
    """
    N = scene.primal_points.shape[0]
    offsets = scene.point_adjacency_offsets.to(torch.long)
    neighs  = scene.point_adjacency.to(torch.long)

    deg = offsets[1:] - offsets[:-1]   # [N] degree per point
    max_deg = deg.max().item()

    # expanded node indices aligned with neighbor list
    expanded_nodes = torch.arange(N, device=scene.device).repeat_interleave(deg)
    # relative position of each neighbor
    rel_pos = torch.arange(neighs.numel(), device=scene.device) - offsets[expanded_nodes]

    # scatter into padded matrix
    out = torch.full((N, max_deg), pad_value, device=scene.device, dtype=torch.long)
    out[expanded_nodes, rel_pos] = neighs
    return out