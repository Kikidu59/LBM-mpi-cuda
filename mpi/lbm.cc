#include "lbm.hh"

#include <algorithm>

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

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r,
         MPI_Comm comm)
  : nx_(nx), ny_(ny), ny_loc_(0), nyt_(0), y0_(0),
    u_in_(u_in), tau_(0.0),
    comm_(comm), up_(MPI_PROC_NULL), down_(MPI_PROC_NULL)
{
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // My rank, the communicator size, and the up/down neighbour ranks along y
  // (MPI_PROC_NULL on the first/last rank, so those exchanges are no-ops).
  MPI_Comm_rank(comm_, &prank_);
  MPI_Comm_size(comm_, &psize_);
  down_ = (prank_ > 0)          ? prank_ - 1 : MPI_PROC_NULL;  // smaller y
  up_   = (prank_ < psize_ - 1)  ? prank_ + 1 : MPI_PROC_NULL;  // larger  y

  // 1D decomposition along y: each rank owns a contiguous block of rows. The
  // first 'rem' ranks take one extra row so the load stays balanced.
  const std::size_t base = ny_ / std::size_t(psize_);
  const std::size_t rem  = ny_ % std::size_t(psize_);
  const std::size_t r    = std::size_t(prank_);
  ny_loc_ = base + (r < rem ? 1 : 0);
  y0_     = r * base + std::min(r, rem);   // global y of my first real row
  nyt_    = ny_loc_ + 2;                    // + bottom halo + top halo

  // Local arrays: real rows plus one halo (ghost) row on each side.
  f_.assign   (9 * nx_ * nyt_, 0.0);
  ftmp_.assign(9 * nx_ * nyt_, 0.0);
  solid_.assign(nx_ * nyt_, 0);

  // No-slip top and bottom walls. They only exist on the ranks that own the
  // first (y=0) and last (y=ny-1) global rows.
  if (owns_row(0)) {
    for (std::size_t x = 0; x < nx_; ++x) solid_[lidx(x, g2l(0))] = 1;
  }
  if (owns_row(ny_ - 1)) {
    for (std::size_t x = 0; x < nx_; ++x) solid_[lidx(x, g2l(ny_ - 1))] = 1;
  }

  mark_obstacle(cyl_x, cyl_y, cyl_r);
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
  // Loop over my real rows only, using the global y for the distance test.
  for (std::size_t L = 1; L <= ny_loc_; ++L) {
    const std::size_t gy = y0_ + (L - 1);            // global y of this row
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x)  - c_x;
      const double dy = double(gy) - c_y;
      if (dx * dx + dy * dy <= r2) solid_[lidx(x, L)] = 1;
    }
  }
}

void
LBM::initialize()
{
  // Initialize the real rows only (local rows 1..ny_loc).
  for (std::size_t L = 1; L <= ny_loc_; ++L) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double rho = 1.0;
      const double ux  = solid_[lidx(x, L)] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(i, x, L)] = feq;
      }
    }
  }
}

void
LBM::step()
{
  collide();
  bounce_back();
  halo_exchange();   // refresh the ghost rows before the neighbour read
  stream();
  apply_inlet();
  apply_outlet();
}

void
LBM::collide()
{
  const double inv_tau = 1.0 / tau_;
  // Loop over the local real rows; index with fidx (local coordinates).
  for (std::size_t L = 1; L <= ny_loc_; ++L) {
    for (std::size_t x = 0; x < nx_; ++x) {
      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[fidx(i, x, L)];
        rho += fi;
        mx  += cx[i] * fi;
        my  += cy[i] * fi;
      }
      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(i, x, L)] += -inv_tau * (f_[fidx(i, x, L)] - feq);
      }
    }
  }
}

void
LBM::bounce_back()
{
  // Fullway bounce-back: in solid cells, swap each pair of opposite directions.
  // Combined with subsequent streaming this reflects populations across the
  // solid-fluid interface.
  for (std::size_t L = 1; L <= ny_loc_; ++L) {
    for (std::size_t x = 0; x < nx_; ++x) {
      if (!solid_[lidx(x, L)]) continue;
      std::swap(f_[fidx(1, x, L)], f_[fidx(3, x, L)]);
      std::swap(f_[fidx(2, x, L)], f_[fidx(4, x, L)]);
      std::swap(f_[fidx(5, x, L)], f_[fidx(7, x, L)]);
      std::swap(f_[fidx(6, x, L)], f_[fidx(8, x, L)]);
    }
  }
}

// Halo exchange. Send my top real row to the up-neighbour (it stores it as its
// bottom halo) and at the same time receive my bottom halo from the
// down-neighbour; then the symmetric exchange for the other side. All 9
// directions of a boundary row are packed into one contiguous buffer, so each
// side is a single message. MPI_Sendrecv cannot deadlock, and MPI_PROC_NULL
// neighbours are no-ops.
void
LBM::halo_exchange()
{
  if (psize_ == 1) return;   // nothing to exchange when running on one rank

  std::vector<double> sbuf(9 * nx_), rbuf(9 * nx_);
  MPI_Status status;

  // --- exchange 1: top real row -> up ; bottom halo <- down ---
  for (int i = 0; i < Q; ++i)
    std::copy(&f_[fidx(i, 0, ny_loc_)], &f_[fidx(i, 0, ny_loc_)] + nx_, &sbuf[i * nx_]);
  MPI_Sendrecv(sbuf.data(), int(9 * nx_), MPI_DOUBLE, up_,   0,
               rbuf.data(), int(9 * nx_), MPI_DOUBLE, down_, 0,
               comm_, &status);
  for (int i = 0; i < Q; ++i)
    std::copy(&rbuf[i * nx_], &rbuf[i * nx_] + nx_, &f_[fidx(i, 0, 0)]);

  // --- exchange 2: bottom real row -> down ; top halo <- up ---
  for (int i = 0; i < Q; ++i)
    std::copy(&f_[fidx(i, 0, 1)], &f_[fidx(i, 0, 1)] + nx_, &sbuf[i * nx_]);
  MPI_Sendrecv(sbuf.data(), int(9 * nx_), MPI_DOUBLE, down_, 1,
               rbuf.data(), int(9 * nx_), MPI_DOUBLE, up_,   1,
               comm_, &status);
  for (int i = 0; i < Q; ++i)
    std::copy(&rbuf[i * nx_], &rbuf[i * nx_] + nx_, &f_[fidx(i, 0, nyt_ - 1)]);
}

void
LBM::stream()
{
  // Pull-style streaming: ftmp[i, x, y] = f[i, x - cx[i], y - cy[i]].
  // Boundary cells whose source would be outside the domain keep their
  // current value; the inlet/outlet routines overwrite the relevant ones.
  // The out-of-domain test is done in GLOBAL y so the physical top/bottom
  // edges behave exactly like the serial code (keep own value), while interior
  // rank boundaries read the freshly exchanged halo rows.
  for (int i = 0; i < Q; ++i) {
    for (std::size_t L = 1; L <= ny_loc_; ++L) {
      const std::size_t gy = y0_ + (L - 1);          // global y of this cell
      for (std::size_t x = 0; x < nx_; ++x) {
        const long sx  = long(x)  - cx[i];           // source x (local == global)
        const long gsy = long(gy) - cy[i];           // source y in GLOBAL coords
        if (sx >= 0 && sx < long(nx_) && gsy >= 0 && gsy < long(ny_)) {
          const std::size_t srcL = L - cy[i];        // source row in LOCAL coords
          ftmp_[fidx(i, x, L)] = f_[fidx(i, std::size_t(sx), srcL)];
        } else {
          ftmp_[fidx(i, x, L)] = f_[fidx(i, x, L)];
        }
      }
    }
  }
  f_.swap(ftmp_);
}

void
LBM::apply_inlet()
{
  // Reset the inlet column to the equilibrium distribution corresponding
  // to a prescribed uniform velocity (u_in, 0) and unit density. Simple,
  // stable, and accurate enough for moderate Reynolds numbers.
  // The inlet column exists on every rank (we split along y); real rows only.
  const std::size_t x = 0;
  for (std::size_t L = 1; L <= ny_loc_; ++L) {
    if (solid_[lidx(x, L)]) continue;
    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux * ux + uy * uy;
    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i] * ux + cy[i] * uy;
      f_[fidx(i, x, L)] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM::apply_outlet()
{
  // Zero-gradient outlet: copy the second-to-last column into the last.
  if (nx_ < 2) return;
  const std::size_t x  = nx_ - 1;
  const std::size_t xs = nx_ - 2;
  for (std::size_t L = 1; L <= ny_loc_; ++L) {
    for (int i = 0; i < Q; ++i) {
      f_[fidx(i, x, L)] = f_[fidx(i, xs, L)];
    }
  }
}

// Macroscopic accessors. They take GLOBAL (gx, gy) coordinates and are only
// valid on the rank that owns the row gy.
double
LBM::rho(std::size_t gx, std::size_t gy) const
{
  const std::size_t L = g2l(gy);
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[fidx(i, gx, L)];
  return r;
}

double
LBM::ux(std::size_t gx, std::size_t gy) const
{
  const std::size_t L = g2l(gy);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[fidx(i, gx, L)];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t gx, std::size_t gy) const
{
  const std::size_t L = g2l(gy);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[fidx(i, gx, L)];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

bool
LBM::is_solid(std::size_t gx, std::size_t gy) const
{
  return solid_[lidx(gx, g2l(gy))] != 0;
}
