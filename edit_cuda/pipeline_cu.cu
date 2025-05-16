#include "ATen/TensorIndexing.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <torch/extension.h>
#include <vector>

#define MAX_ITERS_BG 50
#define MAX_ITERS_CHAR 100
#define THREADS_PER_BLOCK 256

// Error check
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  }

// Kernel: assign each pixel to nearest centroid
__global__ void assign_labels_kernel(const float *__restrict__ pixels, int N,
                                     int C, const float *__restrict__ centroids,
                                     const int *__restrict__ sorted_ids, int K,
                                     int *__restrict__ labels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  const float *p = pixels + idx * C;
  float best_dist = 1e30;
  int best_k = 0;
  for (int k = 0; k < K; ++k) {
    const float *c = centroids + k * C;
    float dist = 0;
    for (int d = 0; d < C; ++d) {
      float diff = p[d] - c[d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_k = k;
    }
  }
  labels[idx] = sorted_ids[best_k];
}

// Kernel: partial sums & counts for centroids
__global__ void update_centroids_kernel(const float *__restrict__ pixels, int N,
                                        int C, const int *__restrict__ labels,
                                        float *__restrict__ cent_accum,
                                        int *__restrict__ counts, int K) {
  extern __shared__ float smem[];
  float *sums = smem;                // size K*C
  int *cnts = (int *)(sums + K * C); // size K
  int tid = threadIdx.x;

  // initialize shared memory sums and counts
  for (int i = tid; i < K * C; i += blockDim.x)
    sums[i] = 0.0f;
  if (tid < K)
    cnts[tid] = 0;
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < N) {
    int lbl = labels[idx];
    atomicAdd(&cnts[lbl], 1);
    const float *p = pixels + idx * C;
    for (int d = 0; d < C; ++d) {
      atomicAdd(&sums[lbl * C + d], p[d]);
    }
  }
  __syncthreads();

  // write back to global accumulators
  int total_vals = K * C;
  for (int i = tid; i < total_vals; i += blockDim.x) {
    atomicAdd(&cent_accum[i], sums[i]);
  }
  if (tid < K) {
    atomicAdd(&counts[tid], cnts[tid]);
  }
}

// Generic GPU KMeans with separate accumulator
void kmeans_gpu(at::Tensor pixels, int K, int max_iters, at::Tensor centroids,
                at::Tensor labels) {
  int N = pixels.size(0);
  int C = pixels.size(1);
  int threads = THREADS_PER_BLOCK;
  int blocks = (N + threads - 1) / threads;

  for (int iter = 0; iter < max_iters; ++iter) {
    // create accumulators
    at::Tensor cent_accum = at::zeros_like(centroids);
    at::Tensor counts = at::zeros({K}, centroids.options().dtype(at::kInt));

    auto sorted_id = std::get<1>(centroids.select(1, 1).sort()).to(torch::kInt);
    centroids = centroids.index_select(0, sorted_id);

    // assignment step uses old centroids
    assign_labels_kernel<<<blocks, threads>>>(
        pixels.data_ptr<float>(), N, C, centroids.data_ptr<float>(),
        sorted_id.data_ptr<int>(), K, labels.data_ptr<int>());
    // update step accumulates into cent_accum and counts
    size_t shared_bytes = K * C * sizeof(float) + K * sizeof(int);
    update_centroids_kernel<<<blocks, threads, shared_bytes>>>(
        pixels.data_ptr<float>(), N, C, labels.data_ptr<int>(),
        cent_accum.data_ptr<float>(), counts.data_ptr<int>(), K);
    CUDA_CHECK(cudaDeviceSynchronize());
    // finalize centroids: divide sums by counts
    // Use device tensor operations to avoid host-device pointer issues
    // Create a float tensor of counts with shape [K,1]
    auto counts_f = counts.to(centroids.dtype()).unsqueeze(1); // [K] -> [K,1]
    // Avoid division by zero by clamping
    auto counts_safe = counts_f.clamp_min(1);
    centroids.copy_(cent_accum.div(counts_safe));
  }
}

// Flip label based on border
void flip_label(at::Tensor &label, int H, int W) {
  // auto l = label.view({H, W});
  int halfH = H / 2, halfW = W / 2;
  int cntL = label.index({at::indexing::Slice(), 0}).sum().item<int>();
  int cntR = label.index({at::indexing::Slice(), W - 1}).sum().item<int>();
  int cntT = label.index({0, at::indexing::Slice()}).sum().item<int>();
  int cntB = label.index({H - 1, at::indexing::Slice()}).sum().item<int>();
  int ident = (cntL > halfH) + (cntR > halfH) + (cntT > halfW) + (cntB > halfW);
  if (ident >= 3)
    label = 1 - label;
}

// Main edit function (adjusted for proper coords extraction)
std::vector<at::Tensor> scene_text_edit(at::Tensor images, at::Tensor tokens,
                                        at::Tensor lengths, at::Tensor patches,
                                        at::Tensor edit_ops, int blank,
                                        int charset_start, int token_start,
                                        double p) {
  auto opts = images.options();
  int B = images.size(0), C = images.size(1), H = images.size(2),
      W = images.size(3);
  auto out_images = images.clone();
  auto out_tokens = tokens.clone();
  auto out_lengths = lengths.clone();
  auto max_length = lengths.max().item<int>();
  if (torch::rand({1}).item<double>() > p)
    return {out_images, out_tokens, out_lengths};

  // Flatten pixels for clustering
  auto pixels = images.permute({0, 2, 3, 1}).contiguous().view({B, H * W, C});
  for (int b = 0; b < B; ++b) {
    // --- First KMeans: separate BG/FG ---
    int N = H * W;
    auto pix = pixels[b]; // [N, C]
    // initialize centroids and labels
    int Kbg = 2;
    auto cent_bg = torch::rand({Kbg, C}, opts);
    auto lbl_bg = torch::zeros({N}, images.options().dtype(at::kInt));
    kmeans_gpu(pix.contiguous(), Kbg, MAX_ITERS_BG, cent_bg, lbl_bg);
    // reshape to 2D and flip labels
    auto label2d = lbl_bg.view({H, W});

    // bg_labels2d[b] = label2d;
    flip_label(label2d, H, W);
    // extract 2D coordinates of foreground pixels
    auto coords = torch::nonzero(label2d == 1); // [M, 2]
    if (coords.size(0) < 1)
      continue;

    // --- Second KMeans: character-level clustering ---
    int Kchar = lengths[b].item<int>();
    auto coordf = coords.toType(at::kFloat); // [M,2]
    auto cent_ch = torch::rand({Kchar, 2}, opts);
    cent_ch.select(1, 0) *= H;
    cent_ch.select(1, 1) *= W;
    auto lbl_ch =
        torch::zeros({coords.size(0)}, images.options().dtype(at::kInt));
    kmeans_gpu(coordf.contiguous(), Kchar, MAX_ITERS_CHAR, cent_ch, lbl_ch);
    auto sorted = std::get<1>(cent_ch.select(1, 1).sort());

    // Random edit operation
    int op = edit_ops[torch::randint(0, edit_ops.size(0), {1}).item<int>()]
                 .item<int>();
    int pos = torch::randint(0, Kchar, {1}).item<int>();
    int sel = sorted[pos].item<int>();
    // Select coords of chosen character cluster
    auto sel_coords = coords.index({lbl_ch == sel}); // [m,2]
    if (sel_coords.size(0) < 2)
      continue;

    int y1 = sel_coords.index({at::indexing::Slice(), 0}).min().item<int>();
    int y2 = sel_coords.index({at::indexing::Slice(), 0}).max().item<int>();
    int x1 = sel_coords.index({at::indexing::Slice(), 1}).min().item<int>();
    int x2 = sel_coords.index({at::indexing::Slice(), 1}).max().item<int>();
    int ph = y2 - y1, pw = x2 - x1;
    if (ph <= 0 || pw < 2)
      continue;

    auto region =
        out_images[b].index({at::indexing::Slice(), at::indexing::Slice(y1, y2),
                             at::indexing::Slice(x1, x2)});
    auto bg = region.mean({1, 2}, true);
    int max_len = tokens.size(1);
    int len = lengths[b].item<int>();
    pos += token_start;
    int tok = tokens[b][pos].item<int>();
    switch (op) {
    case 0: {
      if (len > 1) {
        out_images[b].index_put_({at::indexing::Slice(),
                                  at::indexing::Slice(y1, y2),
                                  at::indexing::Slice(x1, x2)},
                                 bg);
        auto t = out_tokens[b];
        t = torch::cat({t.slice(0, 0, pos), t.slice(0, pos + 1),
                        torch::full({1}, blank,
                                    opts.device(t.device()).dtype(t.dtype()))});
        out_tokens[b] = t;
        out_lengths[b] = len - 1;
      }
    } break;
    case 1: {
      int rt;
      do {
        rt = torch::randint(charset_start, patches.size(0) + charset_start, {1})
                 .item<int>();
      } while (rt == tok);
      auto pch = torch::nn::functional::interpolate(
                     patches[rt - charset_start].unsqueeze(0),
                     torch::nn::functional::InterpolateFuncOptions()
                         .size(std::vector<int64_t>{ph, pw})
                         .mode(torch::kBilinear)
                         .align_corners(false))
                     .squeeze(0);
      out_images[b].index_put_({at::indexing::Slice(),
                                at::indexing::Slice(y1, y2),
                                at::indexing::Slice(x1, x2)},
                               torch::clamp(bg + pch, 0, 1));
      auto t = out_tokens[b];
      t[pos] = rt;
      out_tokens[b] = t;
    } break;
    case 2: {
      if (len < max_length) {
        int rt =
            torch::randint(charset_start, patches.size(0) + charset_start, {1})
                .item<int>();
        auto p1 = torch::nn::functional::interpolate(
                      patches[tok - charset_start].unsqueeze(0),
                      torch::nn::functional::InterpolateFuncOptions()
                          .size(std::vector<int64_t>{ph, pw / 2})
                          .mode(torch::kBilinear)
                          .align_corners(false))
                      .squeeze(0);
        auto p2 = torch::nn::functional::interpolate(
                      patches[rt - charset_start].unsqueeze(0),
                      torch::nn::functional::InterpolateFuncOptions()
                          .size(std::vector<int64_t>{ph, pw - pw / 2})
                          .mode(torch::kBilinear)
                          .align_corners(false))
                      .squeeze(0);
        out_images[b].index_put_(
            {at::indexing::Slice(), at::indexing::Slice(y1, y2),
             at::indexing::Slice(x1, x2)},
            torch::clamp(bg + torch::cat({p1, p2}, 2), 0, 1));
        auto t = out_tokens[b];
        t = torch::cat({t.slice(0, 0, pos + 1),
                        torch::tensor({rt}, opts.dtype(t.dtype())),
                        t.slice(0, pos + 1, -1)});
        out_tokens[b] = t;
        out_lengths[b] = len + 1;
      }
    } break;
    }
  }
  return {out_images, out_tokens, out_lengths};
}