#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *        MPI version (1D domain decomposition).
 *
 * The solver simulates incompressible flow past one (or two) circular
 * cylinders inside a rectangular channel with no-slip top and bottom walls.
 * The inlet (x = 0) prescribes a uniform horizontal velocity; the outlet
 * (x = nx - 1) is a simple zero-gradient copy from the column to its left.
 * Lattice units are used throughout (dx = dt = 1, c = 1, c_s^2 = 1/3).
 *
 * MPI: the global nx x ny domain is split with a 1D decomposition ALONG y;
 * each rank owns a contiguous block of rows. Rows are contiguous in the
 * structure-of-arrays layout, so the halo rows we exchange are contiguous too,
 * which keeps the communication simple.
 *
 * Local storage adds one "ghost" (halo) row at the bottom and one at the top:
 *   local row 0          = bottom halo  (copy of the down-neighbour's top row)
 *   local rows 1..ny_loc = the real rows owned by this rank
 *   local row ny_loc+1   = top halo     (copy of the up-neighbour's bottom row)
 *
 * Global row gy maps to local row L = gy - y0_ + 1, where y0_ is the global y
 * of this rank's first real row.
 *
 * Per-direction slab stride is nx_ * nyt_ with nyt_ = ny_loc + 2.
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
   * @param comm     MPI communicator the domain is split over.
   */
  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r,
      MPI_Comm comm);

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Set f to the equilibrium distribution with rho = 1, u = (u_in, 0)
  /// on every fluid cell, and (0, 0) on solid cells.
  void initialize();

  /// Advance the simulation by one time step.
  void step();

  // Macroscopic accessors. They take GLOBAL coordinates and are only valid on
  // the rank that owns the row gy (check with owns_row first).
  double rho(std::size_t gx, std::size_t gy) const;
  double ux (std::size_t gx, std::size_t gy) const;
  double uy (std::size_t gx, std::size_t gy) const;
  bool   is_solid(std::size_t gx, std::size_t gy) const;

  /// True if global row gy is a real (non-halo) row on this rank.
  bool owns_row(std::size_t gy) const { return gy >= y0_ && gy < y0_ + ny_loc_; }

  std::size_t nx()       const { return nx_; }
  std::size_t ny()       const { return ny_; }      ///< global ny
  std::size_t ny_local() const { return ny_loc_; }
  double      tau()      const { return tau_; }
  double      u_in()     const { return u_in_; }

private:
  std::size_t lidx (std::size_t x, std::size_t L)        const { return L * nx_ + x; }
  std::size_t fidx (int i, std::size_t x, std::size_t L) const { return i * nx_ * nyt_ + lidx(x, L); }
  std::size_t g2l  (std::size_t gy)                      const { return gy - y0_ + 1; }

  void mark_obstacle (double c_x, double c_y, double r);
  void collide       ();
  void bounce_back   ();
  void halo_exchange ();
  void stream        ();
  void apply_inlet   ();
  void apply_outlet  ();

  std::size_t nx_, ny_;    ///< global grid size
  std::size_t ny_loc_;     ///< number of real rows on this rank
  std::size_t nyt_;        ///< ny_loc_ + 2 (real rows + 2 halo rows)
  std::size_t y0_;         ///< global y of this rank's first real row
  double u_in_;
  double tau_;

  MPI_Comm comm_;
  int prank_, psize_;
  int up_, down_;          ///< neighbour ranks (MPI_PROC_NULL at the edges)

  std::vector<double>  f_;      ///< Current distributions, size 9*nx*nyt.
  std::vector<double>  ftmp_;   ///< Scratch buffer for streaming.
  std::vector<uint8_t> solid_;  ///< 0 = fluid, 1 = solid.  Size nx*nyt.
};

#endif  // LBM_HH
