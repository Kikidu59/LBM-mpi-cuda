#include "lbm.hh"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace {

using Args = std::unordered_map<std::string, std::string>;

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
  const Args kv = parse_args(argc, argv);

  // Grid + physics.
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

  // Threads per block for every kernel (the block/grid study sweeps this).
  const int block = get<int>(kv, "block", 256);

  // Output.
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  // Probe location: about 4 diameters downstream, on the cylinder centerline.
  const std::size_t px = get<std::size_t>(kv, "probe_x",
                                          std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.set_block_size(block);
  solver.initialize();
  solver.enable_probe(px, py, steps);   // device buffer for the (ux,uy) series

  std::cout << "LBM 2D D2Q9 BGK (CUDA)\n"
            << "  grid          : " << nx << " x " << ny << "\n"
            << "  Re            : " << Re << "\n"
            << "  u_in          : " << u_in << "\n"
            << "  tau           : " << solver.tau() << "\n"
            << "  cylinder      : (" << cx0 << ", " << cy0
            << "), r = " << cr0 << "\n";
  if (cr1 > 0.0)
    std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1
              << "), r = " << cr1 << "\n";
  std::cout << "  steps         : " << steps << "\n"
            << "  block size    : " << block << "\n"
            << "  probe at      : (" << px << ", " << py << ")\n"
            << "  probe csv     : " << probe_csv << "\n";

  using clk = std::chrono::high_resolution_clock;

  // Make sure the init copies are finished before we start the timer, then
  // time the pure GPU time loop. Kernel launches are asynchronous, so we
  // synchronize once at the end before reading the clock. We use the same
  // wall-clock + MLUPS definition as the serial/MPI versions for a fair
  // comparison.
  cudaDeviceSynchronize();
  const auto t0 = clk::now();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step();
    solver.record_probe(step - 1);   // GPU write into the per-step slot
  }
  cudaDeviceSynchronize();           // wait for all queued kernels

  const double dt    = std::chrono::duration<double>(clk::now() - t0).count();
  const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;

  // Copy the probe time series back once and write it to disk.
  std::vector<double> probe_data;
  solver.fetch_probe(probe_data);
  std::ofstream probe(probe_csv);
  probe << "step,ux,uy\n";
  for (std::size_t step = 1; step <= steps; ++step) {
    probe << step << ',' << probe_data[2 * (step - 1)] << ','
          << probe_data[2 * (step - 1) + 1] << '\n';
  }

  std::cout << "Wall time : " << dt    << " s\n"
            << "MLUPS     : " << mlups << "\n";
  return 0;
}
