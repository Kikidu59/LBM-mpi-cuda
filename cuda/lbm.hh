#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *
 * The solver simulates incompressible flow past one (or two) circular
 * cylinders inside a rectangular channel with no-slip top and bottom walls.
 * The inlet (x = 0) prescribes a uniform horizontal velocity; the outlet
 * (x = nx - 1) is a simple zero-gradient copy from the column to its left.
 *
 * Distributions are stored in structure-of-arrays layout:
 *   f_[ i * (nx*ny) + y*nx + x ]   for direction i in [0, 9).
 *
 * Lattice units are used throughout (dx = dt = 1, c = 1, c_s^2 = 1/3).
 */
class LBM
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];   ///< Discrete velocity x-components.
  static const int    cy[Q];   ///< Discrete velocity y-components.
  static const double w[Q];    ///< Equilibrium weights.
  static const int    opp[Q];  ///< Index of the opposite direction.

  /**
   * @param nx       Number of cells along x.
   * @param ny       Number of cells along y.
   * @param u_in     Inlet velocity in lattice units (must be << 1/sqrt(3)).
   * @param Re       Target Reynolds number, based on cylinder diameter.
   * @param cyl_x    Center of the (first) cylinder along x, in cell units.
   * @param cyl_y    Center of the (first) cylinder along y, in cell units.
   * @param cyl_r    Radius of the (first) cylinder, in cell units.
   */
  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r);

  // Destructor: free the GPU buffers allocated in the constructor / init.
  ~LBM();

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Set f to the equilibrium distribution with rho = 1, u = (u_in, 0)
  /// on every fluid cell, and (0, 0) on solid cells.
  void initialize();

  /// Advance the simulation by one time step.
  void step();

  // Macroscopic accessors.
  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;

  std::size_t nx()   const { return nx_; }
  std::size_t ny()   const { return ny_; }
  double      tau()  const { return tau_; }
  double      u_in() const { return u_in_; }

  // Controls used by the CUDA driver (main.cc) to configure the GPU run.
  //
  //  set_block_size : number of threads per block for every kernel launch
  //                   (the block/grid study sweeps this value).
  //  enable_probe   : allocate a device buffer that stores (ux,uy) at one
  //                   point for every time step. We write into it on the GPU
  //                   each step (record_probe) and copy it back to the host
  //                   only once at the end (fetch_probe). This keeps the hot
  //                   loop free of any host<->device transfer.
  void set_block_size(int block_size);
  void enable_probe(std::size_t px, std::size_t py, std::size_t nsteps);
  void record_probe(std::size_t step_index);
  void fetch_probe(std::vector<double> & out) const;

private:
  std::size_t idx (std::size_t x, std::size_t y)         const { return y * nx_ + x; }

  // Flat index into a distribution array. SoA (default) keeps each direction
  // contiguous as i*N + k, so that neighbouring cells (neighbouring threads)
  // touch neighbouring addresses -> coalesced memory access on the GPU.
  // With -DLAYOUT_AOS we instead pack the 9 directions of one cell together
  // (k*Q + i); this breaks coalescing and lets us measure its cost (the
  // memory-layout study asked by the project). The device kernels use the
  // matching fidx_dev() defined in lbm.cu, so host init and device kernels
  // always agree on the layout.
#ifdef LAYOUT_AOS
  std::size_t fidx(int i, std::size_t x, std::size_t y) const { return idx(x, y) * Q + i; }
#else
  std::size_t fidx(int i, std::size_t x, std::size_t y) const { return i * nx_ * ny_ + idx(x, y); }
#endif

  void mark_obstacle (double c_x, double c_y, double r);
  void collide       ();
  void bounce_back   ();
  void stream        ();
  void apply_inlet   ();
  void apply_outlet  ();

  std::size_t nx_, ny_;
  double u_in_;
  double tau_;

  std::vector<double>  f_;      ///< Current distributions, size 9*nx*ny.
  // In the CUDA version f_ above lives on the host and is used only to build
  // the initial condition; it is copied once into d_f_ and never touched in
  // the time loop. The streaming scratch buffer (the serial ftmp_) now lives
  // on the device as d_ftmp_, so the host copy is no longer needed.
  std::vector<uint8_t> solid_;  ///< 0 = fluid, 1 = solid.  Size nx*ny.

  // Device-side state. The whole time loop runs on these buffers.
  double *  d_f_     = nullptr;  ///< current distributions on the GPU (9*N)
  double *  d_ftmp_  = nullptr;  ///< streaming scratch on the GPU (double buffer)
  uint8_t * d_solid_ = nullptr;  ///< solid mask on the GPU (N)
  double *  d_probe_ = nullptr;  ///< (ux,uy) time series on the GPU (2*nsteps)
  int       block_size_ = 256;   ///< threads per block (set from the CLI)
  std::size_t probe_px_ = 0;     ///< probe location x
  std::size_t probe_py_ = 0;     ///< probe location y
  std::size_t probe_n_  = 0;     ///< number of recorded steps
};

#endif  // LBM_HH
