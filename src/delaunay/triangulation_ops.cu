#include "triangulation_ops.h"

#include "../utils/common_kernels.cuh"
#include "exact_tree_ops.cuh"

namespace radfoam {

template <typename coord_scalar>
__global__ void
farthest_neighbor_kernel(const Vec3<coord_scalar> *__restrict__ points,
                         const uint32_t *point_adjacency,
                         const uint32_t *point_adjacency_offsets,
                         uint32_t num_points,
                         uint32_t *__restrict__ indices,
                         float *__restrict__ cell_radius) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_points) {
        return;
    }

    Vec3f primal_point = points[i];
    uint32_t point_adjacency_begin = point_adjacency_offsets[i];
    uint32_t point_adjacency_end = point_adjacency_offsets[i + 1];
    uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
    uint32_t farthest_idx = UINT32_MAX;
    float sum_distance = 0.0f;
    float max_distance = 0.0f;

    for (uint32_t i = 0; i < num_faces; ++i) {
        uint32_t opposite_point_idx =
            point_adjacency[point_adjacency_begin + i];
        Vec3f opposite_point = points[opposite_point_idx];

        float distance = (opposite_point - primal_point).norm();
        sum_distance += 0.5 * distance;
        if (distance > max_distance) {
            max_distance = distance;
            farthest_idx = opposite_point_idx;
        }
    }

    indices[i] = farthest_idx;
    cell_radius[i] = sum_distance / num_faces;
}

template <typename coord_scalar>
void farthest_neighbor(const Vec3<coord_scalar> *points,
                       const uint32_t *point_adjacency,
                       const uint32_t *point_adjacency_offsets,
                       uint32_t num_points,
                       uint32_t *indices,
                       float *cell_radius,
                       const void *stream) {
    launch_kernel_1d<1024>(farthest_neighbor_kernel<coord_scalar>,
                           num_points,
                           stream,
                           points,
                           point_adjacency,
                           point_adjacency_offsets,
                           num_points,
                           indices,
                           cell_radius);
}

void farthest_neighbor(ScalarType coord_scalar_type,
                       const void *points,
                       const void *point_adjacency,
                       const void *point_adjacency_offsets,
                       uint32_t num_points,
                       void *indices,
                       void *cell_radius,
                       const void *stream) {

    if (coord_scalar_type == ScalarType::Float32) {
        farthest_neighbor(
            static_cast<const Vec3<float> *>(points),
            static_cast<const uint32_t *>(point_adjacency),
            static_cast<const uint32_t *>(point_adjacency_offsets),
            num_points,
            static_cast<uint32_t *>(indices),
            static_cast<float *>(cell_radius),
            stream);
    } else {
        throw std::runtime_error("unsupported scalar type");
    }
}

void dface_area(const void *tets_in,
                const void *tet_adjacency_in,
                const void *edges_in,
                const void *circumcenters_in,
                uint32_t num_tets,
                uint32_t num_edges,
                void *dface_area_out) {
    const IndexedTet *tets = reinterpret_cast<const IndexedTet *>(tets_in);
    const uint32_t *tet_adjacency =
        reinterpret_cast<const uint32_t *>(tet_adjacency_in);
    const IndexedEdge *edges = reinterpret_cast<const IndexedEdge *>(edges_in);
    const Vec3f *circumcenters =
        reinterpret_cast<const Vec3f *>(circumcenters_in);
    float *_dface_area = reinterpret_cast<float *>(dface_area_out);

    CUDAArray<uint32_t> dface_cardinality(num_edges);
    auto dface_cardinality_begin = dface_cardinality.begin();
    cuda_check(cuMemsetD32((CUdeviceptr)dface_cardinality_begin, 0, num_edges));
    CUDAArray<Vec3f> dface_centroid(num_edges);
    auto dface_centroid_begin = dface_centroid.begin();
    cuda_check(cuMemsetD8(
        (CUdeviceptr)dface_centroid_begin, 0, num_edges * sizeof(Vec3f)));
    CUDAArray<uint32_t> dface_to_tet(num_edges);
    auto dface_to_tet_begin = dface_to_tet.begin();

    auto get_dface_cardinality = [=] __device__(uint32_t i) {
        IndexedTet tet = tets[i];
        Vec3f circumcenter = circumcenters[i];
        IndexedEdge tet_edges[6] = {{tet.vertices[0], tet.vertices[1]},
                                    {tet.vertices[0], tet.vertices[2]},
                                    {tet.vertices[0], tet.vertices[3]},
                                    {tet.vertices[1], tet.vertices[2]},
                                    {tet.vertices[1], tet.vertices[3]},
                                    {tet.vertices[2], tet.vertices[3]}};

#pragma unroll
        for (uint32_t j = 0; j < 6; ++j) {
            IndexedEdge edge = tet_edges[j];
            auto it = binary_search(edges, edges + num_edges, edge);
            if (it == edges + num_edges) {
                assert(0); // Edge not found
            }
            size_t t = it - edges;

            uint32_t old_value =
                atomicAdd(dface_cardinality_begin + t, (uint32_t)1);
            atomic_add_vec(dface_centroid_begin + t, circumcenter);
            if (old_value == 0) {
                dface_to_tet_begin[t] = i;
            }
        }
    };
    for_n(u32zero(), num_tets, get_dface_cardinality);

    auto normalize_dface_centroid = [=] __device__(uint32_t i) {
        uint32_t cardinality = dface_cardinality_begin[i];
        if (cardinality > 2) {
            dface_centroid_begin[i] =
                dface_centroid_begin[i] / (float)cardinality;
        } else {
            dface_centroid_begin[i] = Vec3f(0.0f, 0.0f, 0.0f); // Convex hull
        }
    };
    for_n(u32zero(), num_edges, normalize_dface_centroid);

    auto compute_dface_area = [=] __device__(uint32_t i) {
        IndexedEdge edge = edges[i];
        uint32_t cardinality = dface_cardinality_begin[i];
        if (cardinality < 3) {
            _dface_area[i] = 0.0f; // Convex hull
            return;
        }
        float area = 0.0f;

        Vec3f centroid = dface_centroid_begin[i];
        uint32_t tet_idx = dface_to_tet_begin[i];
        uint32_t prev_tet_idx = UINT32_MAX;
        bool done = false;
        uint32_t walked_through = 0;

        while (!done) {
            walked_through++;
            IndexedTet curr_tet = tets[tet_idx];

            if (walked_through > cardinality) {
                printf("Infinite loop detected in dface area computation of "
                       "edge %d because walked through %d, expected %d\n",
                       i,
                       walked_through,
                       cardinality);
                assert(0);
            }

            bool found = false;
            for (uint32_t k = 0; k < 4; ++k) {
                auto face = curr_tet.face(k);
                auto neighbor_idx = tet_adjacency[tet_idx * 4 + k];
                if (neighbor_idx == UINT32_MAX) {
                    continue; // Convex hull
                } else if ((face.edge(0) == edge || face.edge(1) == edge ||
                            face.edge(2) == edge)) {
                    neighbor_idx /= 4;
                    if (neighbor_idx != prev_tet_idx) {
                        Vec3f ab = circumcenters[tet_idx] - centroid;
                        Vec3f ac = circumcenters[neighbor_idx] - centroid;
                        area += 0.5f * (ab.cross(ac)).norm();

                        prev_tet_idx = tet_idx;
                        tet_idx = neighbor_idx;
                        found = true;
                        if (tet_idx == dface_to_tet_begin[i]) {
                            done = true; // Completed the loop
                        }
                        break;
                    }
                }
            }
            if (!found) {
                _dface_area[i] = 0.0f; // Convex hull
                return;
            }
        }
        if (walked_through != cardinality) {
            printf("Walked through %d, expected %d for edge %d\n",
                   walked_through,
                   cardinality,
                   i);
            assert(0);
        }
        _dface_area[i] = area;
    };
    for_n(u32zero(), num_edges, compute_dface_area);
}

} // namespace radfoam