#include "lbm.hh"

#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Small helper: abort with a clear message if a CUDA call fails.
#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::exit(EXIT_FAILURE);                                \
    }                                                         \
  } while (0)

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S
const int LBM::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// Device copies of the lattice constants, placed in constant memory. Constant
// memory is cached and broadcast to all threads of a warp in one access, which
// is ideal here because every thread reads the same cx/cy/w values. We use
// static initializers so no explicit cudaMemcpyToSymbol is needed.
__constant__ int    d_cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int    d_cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ double d_w[9]  = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

// Device flat index, matching LBM::fidx in the header (same layout choice).
// k = y*nx + x is the linear cell index; N = nx*ny is the number of cells.
#ifdef LAYOUT_AOS
__device__ inline std::size_t fidx_dev(int i, std::size_t k, std::size_t /*N*/) { return k * 9 + i; }
#else
__device__ inline std::size_t fidx_dev(int i, std::size_t k, std::size_t N)     { return i * N + k; }
#endif

// One thread per cell. Each kernel below is the GPU version of one method of
// the serial solver; the algorithm is line-for-line the same, only the loop
// over cells is replaced by "one thread = one cell".

// collide(): BGK relaxation toward the local equilibrium. Fully local.
__global__ void collide_kernel(double * f, std::size_t N, double inv_tau)
{
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;

  double rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    const double fi = f[fidx_dev(i, k, N)];
    rho += fi;
    mx  += d_cx[i] * fi;
    my  += d_cy[i] * fi;
  }
  const double ux = (rho > 0.0) ? mx / rho : 0.0;
  const double uy = (rho > 0.0) ? my / rho : 0.0;
  const double u2 = ux * ux + uy * uy;

  for (int i = 0; i < 9; ++i) {
    const double cu  = d_cx[i] * ux + d_cy[i] * uy;
    const double feq = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    const std::size_t fk = fidx_dev(i, k, N);
    f[fk] += -inv_tau * (f[fk] - feq);
  }
}

// bounce_back(): in solid cells, swap each pair of opposite directions.
__global__ void bounce_back_kernel(double * f, const uint8_t * solid, std::size_t N)
{
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  if (!solid[k]) return;

  double t;
  t = f[fidx_dev(1, k, N)]; f[fidx_dev(1, k, N)] = f[fidx_dev(3, k, N)]; f[fidx_dev(3, k, N)] = t;
  t = f[fidx_dev(2, k, N)]; f[fidx_dev(2, k, N)] = f[fidx_dev(4, k, N)]; f[fidx_dev(4, k, N)] = t;
  t = f[fidx_dev(5, k, N)]; f[fidx_dev(5, k, N)] = f[fidx_dev(7, k, N)]; f[fidx_dev(7, k, N)] = t;
  t = f[fidx_dev(6, k, N)]; f[fidx_dev(6, k, N)] = f[fidx_dev(8, k, N)]; f[fidx_dev(8, k, N)] = t;
}

// stream(): pull-style, ftmp[i,k] = f[i, source(k)]. Reads neighbours of f and
// writes its own cell of ftmp, so there is no race (double buffering). Cells
// whose source is outside the domain keep their own value (edges are then
// overwritten by the inlet/outlet kernels), exactly like the serial code.
__global__ void stream_kernel(const double * f, double * ftmp, std::size_t nx, std::size_t ny)
{
  const std::size_t N = nx * ny;
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;

  const long x = long(k % nx);
  const long y = long(k / nx);
  for (int i = 0; i < 9; ++i) {
    const long sx = x - d_cx[i];
    const long sy = y - d_cy[i];
    if (sx >= 0 && sx < long(nx) && sy >= 0 && sy < long(ny)) {
      const std::size_t ks = std::size_t(sy) * nx + std::size_t(sx);
      ftmp[fidx_dev(i, k, N)] = f[fidx_dev(i, ks, N)];
    } else {
      ftmp[fidx_dev(i, k, N)] = f[fidx_dev(i, k, N)];
    }
  }
}

// apply_inlet(): one thread per row, reset the x=0 column to equilibrium at
// the prescribed uniform velocity (u_in, 0) and unit density.
__global__ void inlet_kernel(double * f, const uint8_t * solid,
                             std::size_t nx, std::size_t ny, double u_in)
{
  const std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;

  const std::size_t N = nx * ny;
  const std::size_t k = y * nx + 0;
  if (solid[k]) return;

  const double rho = 1.0;
  const double ux  = u_in;
  const double uy  = 0.0;
  const double u2  = ux * ux + uy * uy;
  for (int i = 0; i < 9; ++i) {
    const double cu = d_cx[i] * ux + d_cy[i] * uy;
    f[fidx_dev(i, k, N)] = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
  }
}

// apply_outlet(): one thread per row, zero-gradient copy of column nx-2 into
// the last column nx-1.
__global__ void outlet_kernel(double * f, std::size_t nx, std::size_t ny)
{
  const std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;

  const std::size_t N  = nx * ny;
  const std::size_t k  = y * nx + (nx - 1);
  const std::size_t ks = y * nx + (nx - 2);
  for (int i = 0; i < 9; ++i) {
    f[fidx_dev(i, k, N)] = f[fidx_dev(i, ks, N)];
  }
}

// Probe: a single thread reads the 9 populations at one cell and writes its
// (ux, uy) into out[0], out[1]. Launched once per step into a per-step slot of
// the device probe buffer, so no host<->device copy happens inside the loop.
__global__ void probe_kernel(const double * f, std::size_t nx, std::size_t ny,
                             std::size_t px, std::size_t py, double * out)
{
  const std::size_t N = nx * ny;
  const std::size_t k = py * nx + px;
  double rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    const double fi = f[fidx_dev(i, k, N)];
    rho += fi;
    mx  += d_cx[i] * fi;
    my  += d_cy[i] * fi;
  }
  out[0] = (rho > 0.0) ? mx / rho : 0.0;
  out[1] = (rho > 0.0) ? my / rho : 0.0;
}

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r)
  : nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0),
    f_   (9 * nx * ny, 0.0),
    solid_(nx * ny, 0)
{
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // No-slip top and bottom walls.
  for (std::size_t x = 0; x < nx_; ++x) {
    solid_[idx(x, 0)]        = 1;
    solid_[idx(x, ny_ - 1)]  = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);

  // Allocate the device buffers (sizes are known from nx, ny). They are filled
  // later in initialize(), after any second cylinder has been added.
  const std::size_t N = nx_ * ny_;
  CUDA_CHECK(cudaMalloc(&d_f_,     9 * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ftmp_,  9 * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_solid_, N * sizeof(uint8_t)));
}

LBM::~LBM()
{
  // Free everything we allocated. cudaFree(nullptr) is a no-op, so this is safe
  // even if enable_probe() was never called.
  cudaFree(d_f_);
  cudaFree(d_ftmp_);
  cudaFree(d_solid_);
  cudaFree(d_probe_);
}

void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) solid_[idx(x, y)] = 1;
    }
  }
}

void
LBM::initialize()
{
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double rho = 1.0;
      const double ux  = solid_[idx(x, y)] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(i, x, y)] = feq;
      }
    }
  }

  // Copy the initial condition and the solid mask to the GPU. This is the only
  // host->device transfer of the whole run (apart from the constants, which
  // are static-initialized in constant memory).
  const std::size_t N = nx_ * ny_;
  CUDA_CHECK(cudaMemcpy(d_f_,     f_.data(),     9 * N * sizeof(double),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_solid_, solid_.data(), N * sizeof(uint8_t),     cudaMemcpyHostToDevice));
}

void
LBM::step()
{
  collide();
  bounce_back();
  stream();
  // Double buffering: stream() wrote the result into d_ftmp_. Swap the two
  // device pointers so d_f_ now points at the streamed data (no data copy).
  std::swap(d_f_, d_ftmp_);
  apply_inlet();
  apply_outlet();
}

// The five methods below replace the serial CPU loops by a kernel launch. All
// kernels run in the default stream, so they execute in order and each one
// sees the previous one's result without an explicit synchronization.

void
LBM::collide()
{
  const std::size_t N = nx_ * ny_;
  const int grid = int((N + block_size_ - 1) / block_size_);
  collide_kernel<<<grid, block_size_>>>(d_f_, N, 1.0 / tau_);
}

void
LBM::bounce_back()
{
  // Fullway bounce-back: in solid cells, swap each pair of opposite directions.
  // Combined with subsequent streaming this reflects populations across the
  // solid-fluid interface.
  const std::size_t N = nx_ * ny_;
  const int grid = int((N + block_size_ - 1) / block_size_);
  bounce_back_kernel<<<grid, block_size_>>>(d_f_, d_solid_, N);
}

void
LBM::stream()
{
  // Pull-style streaming: ftmp[i, x, y] = f[i, x - cx[i], y - cy[i]].
  // Boundary cells whose source would be outside the domain keep their
  // current value; the inlet/outlet routines overwrite the relevant ones.
  const std::size_t N = nx_ * ny_;
  const int grid = int((N + block_size_ - 1) / block_size_);
  stream_kernel<<<grid, block_size_>>>(d_f_, d_ftmp_, nx_, ny_);
}

void
LBM::apply_inlet()
{
  // Reset the inlet column to the equilibrium distribution corresponding
  // to a prescribed uniform velocity (u_in, 0) and unit density. Simple,
  // stable, and accurate enough for moderate Reynolds numbers.
  const int grid = int((ny_ + block_size_ - 1) / block_size_);
  inlet_kernel<<<grid, block_size_>>>(d_f_, d_solid_, nx_, ny_, u_in_);
}

void
LBM::apply_outlet()
{
  // Zero-gradient outlet: copy the second-to-last column into the last.
  if (nx_ < 2) return;
  const int grid = int((ny_ + block_size_ - 1) / block_size_);
  outlet_kernel<<<grid, block_size_>>>(d_f_, nx_, ny_);
}

void
LBM::set_block_size(int block_size)
{
  block_size_ = block_size;
}

void
LBM::enable_probe(std::size_t px, std::size_t py, std::size_t nsteps)
{
  probe_px_ = px;
  probe_py_ = py;
  probe_n_  = nsteps;
  CUDA_CHECK(cudaMalloc(&d_probe_, 2 * nsteps * sizeof(double)));
}

void
LBM::record_probe(std::size_t step_index)
{
  // One single-thread kernel writes (ux,uy) of the probe cell into the slot of
  // this step. No host<->device copy here: the data stays on the GPU until the
  // end of the run.
  probe_kernel<<<1, 1>>>(d_f_, nx_, ny_, probe_px_, probe_py_,
                         d_probe_ + 2 * step_index);
}

void
LBM::fetch_probe(std::vector<double> & out) const
{
  // Single device->host copy of the whole time series, called once after the
  // time loop.
  out.resize(2 * probe_n_);
  CUDA_CHECK(cudaMemcpy(out.data(), d_probe_, 2 * probe_n_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
}

double
LBM::rho(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_;
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[i * N + idx(x, y)];
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x, y)];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x, y)];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  return solid_[idx(x, y)] != 0;
}
