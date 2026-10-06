#include "lbm.hh"

#include <mpi.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using Args = std::unordered_map<std::string, std::string>;

// Argument parsing is unchanged from the serial version.
Args
parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      std::exit(1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T
get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v; iss >> v;
  return v;
}

std::string
get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}  // namespace

int
main(int argc, char ** argv)
{
  // Initialise MPI and get this process's rank and the total number of ranks.
  MPI_Init(&argc, &argv);
  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  const Args kv = parse_args(argc, argv);

  // Grid + physics (same keys as the serial code).
  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  // Cylinder geometry. Defaults: at (nx/4, ny/2) with radius ny/40
  // (i.e. cylinder diameter = ny/20, ~5% blockage). At Re = 100 this
  // setup reproduces the classical Strouhal number St ~ 0.16.
  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  // Optional second cylinder. Disabled if cyl2_r <= 0.
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  // Probe location: about 4 diameters downstream, on the cylinder centerline.
  const std::size_t px = get<std::size_t>(kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  // Build the solver on its local sub-domain.
  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0, MPI_COMM_WORLD);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  // Only rank 0 prints the run configuration.
  if (prank == 0) {
    std::cout << "LBM 2D D2Q9 BGK  (MPI, " << psize << " ranks)\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  Re            : " << Re << "\n"
              << "  u_in          : " << u_in << "\n"
              << "  tau           : " << solver.tau() << "\n"
              << "  cylinder      : (" << cx0 << ", " << cy0 << "), r = " << cr0 << "\n"
              << "  steps         : " << steps << "\n"
              << "  probe at      : (" << px << ", " << py << ")\n"
              << "  probe csv     : " << probe_csv << "\n";
  }

  // Only the rank that owns the probe row writes probe.csv.
  const bool i_own_probe = solver.owns_row(py);
  std::ofstream probe;
  if (i_own_probe) {
    probe.open(probe_csv);
    probe << "step,ux,uy\n";
  }

  // Timing with MPI_Wtime, synchronised by a barrier.
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step();
    // The probe owner records its point each step.
    if (i_own_probe) {
      probe << step << ',' << solver.ux(px, py) << ',' << solver.uy(px, py) << '\n';
    }
  }

  // Take the slowest rank's time, then compute the global MLUPS.
  MPI_Barrier(MPI_COMM_WORLD);
  const double dt_local = MPI_Wtime() - t0;
  double dt = 0.0;
  MPI_Reduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (prank == 0) {
    const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;
    std::cout << "Wall time : " << dt    << " s\n"
              << "MLUPS     : " << mlups << "\n";
  }

  MPI_Finalize();
  return 0;
}
