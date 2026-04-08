#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "particle.h"
#include "init.h"
#include "forces.h"
#include "integrator.h"
#include "diagnostics.h"
#include "neighbour.h"

// ============================================================
//  TEST 1: Free fall — single particle, compare to analytical
// ============================================================
void test_freefall()
{
    std::cout << "\n=== TEST 1: Free Fall ===\n";
    const double g    = 9.81;
    const double z0   = 10.0;
    const double dt   = 1e-4;
    const double tmax = 1.0;

    Particle p;
    p.x = 0.5; p.y = 0.5; p.z = z0;
    p.vx = 0; p.vy = 0; p.vz = 0;
    p.mass = 1.0; p.radius = 0.05;

    std::ofstream f("test_freefall.csv");
    f << "t,z_num,z_ana,vz_num,vz_ana\n";

    double t = 0.0;
    int step = 0;
    while (t <= tmax) {
        double z_ana  = z0 - 0.5 * g * t * t;
        double vz_ana = -g * t;
        f << std::fixed << std::setprecision(6)
          << t << "," << p.z << "," << z_ana << ","
          << p.vz << "," << vz_ana << "\n";

        // Apply gravity and integrate
        p.fz = p.mass * (-g);
        p.fx = 0; p.fy = 0;
        p.vz += (p.fz / p.mass) * dt;
        p.z  += p.vz * dt;

        t += dt; ++step;
    }
    std::cout << "  Output: test_freefall.csv\n";
}

// ============================================================
//  TEST 2: Constant velocity (g=0)
// ============================================================
void test_constant_velocity()
{
    std::cout << "\n=== TEST 2: Constant Velocity ===\n";
    const double dt   = 1e-4;
    const double tmax = 1.0;

    Particle p;
    p.x = 0.0; p.y = 0.0; p.z = 5.0;
    p.vx = 2.0; p.vy = 1.0; p.vz = 0.5;
    p.mass = 1.0; p.radius = 0.05;

    std::ofstream f("test_const_vel.csv");
    f << "t,x,y,z,x_ana,y_ana,z_ana\n";

    double t = 0.0;
    double x0=p.x, y0=p.y, z0=p.z;
    double vx0=p.vx, vy0=p.vy, vz0=p.vz;

    while (t <= tmax) {
        f << std::fixed << std::setprecision(6)
          << t << ","
          << p.x << "," << p.y << "," << p.z << ","
          << x0+vx0*t << "," << y0+vy0*t << "," << z0+vz0*t << "\n";

        // No gravity, no forces
        p.fx = 0; p.fy = 0; p.fz = 0;
        p.vx += 0; p.vy += 0; p.vz += 0;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        t += dt;
    }
    std::cout << "  Output: test_const_vel.csv\n";
}

// ============================================================
//  TEST 3: Bouncing particle
// ============================================================
void test_bounce()
{
    std::cout << "\n=== TEST 3: Bouncing Particle ===\n";
    const double g     = 9.81;
    const double kn    = 1e5;
    const double gamma = 50.0;
    const double dt    = 1e-5;
    const double tmax  = 3.0;
    const double R     = 0.05;

    Particle p;
    p.x = 0.5; p.y = 0.5; p.z = 1.0;
    p.vx = 0; p.vy = 0; p.vz = 0;
    p.mass = 1.0; p.radius = R;

    std::ofstream f("test_bounce.csv");
    f << "t,z,vz\n";

    double t = 0.0;
    int out_interval = 100;
    int step = 0;

    while (t <= tmax) {
        if (step % out_interval == 0)
            f << std::fixed << std::setprecision(6)
              << t << "," << p.z << "," << p.vz << "\n";

        // Gravity
        p.fz = p.mass * (-g);
        p.fx = 0; p.fy = 0;

        // Floor contact
        double delta = R - p.z;
        if (delta > 0.0) {
            double vn = -p.vz;
            double Fn = std::max(0.0, kn*delta - gamma*vn);
            p.fz += Fn;
        }

        // Integrate
        p.vz += (p.fz / p.mass) * dt;
        p.z  += p.vz * dt;

        t += dt; ++step;
    }
    std::cout << "  Output: test_bounce.csv\n";
}

// ============================================================
//  TIMESTEP CONVERGENCE STUDY
// ============================================================
void test_timestep_convergence()
{
    std::cout << "\n=== Timestep Convergence Study ===\n";
    const double g    = 9.81;
    const double z0   = 5.0;
    const double tmax = 1.0;

    std::ofstream f("test_convergence.csv");
    f << "dt,error_z,error_vz\n";

    for (double dt : {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4}) {
        double z  = z0, vz = 0.0;
        double t  = 0.0;
        while (t < tmax - 1e-12) {
            double fz = -g;
            vz += fz * dt;
            z  += vz * dt;
            t  += dt;
        }
        double z_ana  = z0 - 0.5*g*tmax*tmax;
        double vz_ana = -g*tmax;
        f << std::scientific << dt << ","
          << std::abs(z - z_ana) << ","
          << std::abs(vz - vz_ana) << "\n";
        std::cout << "  dt=" << dt << "  err_z=" << std::abs(z-z_ana) << "\n";
    }
    std::cout << "  Output: test_convergence.csv\n";
}

// ============================================================
//  MAIN SIMULATION (serial, brute force)
// ============================================================
void run_simulation_serial(int N, double tmax, const std::string& tag)
{
    std::cout << "\n=== Serial Simulation N=" << N << " ===\n";

    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    const double R  = 0.03, mass = 1.0;
    const double kn = 1e5, gamma = 50.0;
    const double g  = 9.81;
    const double dt = 1e-5;

    ContactParams cp{kn, gamma, 0.0, 0.0, -g, Lx, Ly, Lz};

    std::vector<Particle> particles;
    initialize_particles(particles, N, Lx, Ly, Lz, R, mass);

    std::cout << "  Actual particle count: " << particles.size() << "\n";

    std::ofstream ke_file("ke_" + tag + ".csv");
    ke_file << "t,KE\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    int out_interval = (int)(0.01 / dt);

    // Timing breakdown
    double time_contacts = 0.0, time_walls = 0.0, time_integrate = 0.0;

    while (t <= tmax) {
        if (step % out_interval == 0) {
            double ke = compute_kinetic_energy(particles);
            ke_file << std::fixed << std::setprecision(6) << t << "," << ke << "\n";
        }

        zero_forces(particles);
        add_gravity(particles, cp);

        auto t0 = std::chrono::high_resolution_clock::now();
        compute_particle_contacts(particles, cp);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_wall_contacts(particles, cp);
        auto t2 = std::chrono::high_resolution_clock::now();
        integrate(particles, dt);
        auto t3 = std::chrono::high_resolution_clock::now();

        time_contacts  += std::chrono::duration<double>(t1-t0).count();
        time_walls     += std::chrono::duration<double>(t2-t1).count();
        time_integrate += std::chrono::duration<double>(t3-t2).count();

        t += dt; ++step;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "  Total time      : " << total << " s\n";
    std::cout << "  Contacts        : " << time_contacts  << " s ("
              << 100*time_contacts/total  << "%)\n";
    std::cout << "  Wall contacts   : " << time_walls     << " s ("
              << 100*time_walls/total     << "%)\n";
    std::cout << "  Integration     : " << time_integrate << " s ("
              << 100*time_integrate/total << "%)\n";

    // Write timing to file for report
    std::ofstream tf("timing_serial_" + tag + ".csv");
    tf << "function,time_s,percent\n";
    tf << "contacts,"  << time_contacts  << "," << 100*time_contacts/total  << "\n";
    tf << "walls,"     << time_walls     << "," << 100*time_walls/total     << "\n";
    tf << "integrate," << time_integrate << "," << 100*time_integrate/total << "\n";
    tf << "total,"     << total          << ",100\n";
}

// ============================================================
//  MAIN SIMULATION (OpenMP parallel)
// ============================================================
void run_simulation_parallel(int N, double tmax, int nthreads, const std::string& tag)
{
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    std::cout << "\n=== Parallel Simulation N=" << N
              << " threads=" << nthreads << " ===\n";

    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    const double R  = 0.03, mass = 1.0;
    const double kn = 1e5, gamma = 50.0;
    const double g  = 9.81;
    const double dt = 1e-5;

    ContactParams cp{kn, gamma, 0.0, 0.0, -g, Lx, Ly, Lz};

    std::vector<Particle> particles;
    initialize_particles(particles, N, Lx, Ly, Lz, R, mass);
    int actual_N = (int)particles.size();

    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;

    while (t <= tmax) {
        // Zero forces — parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < actual_N; ++i) {
            particles[i].fx = 0.0;
            particles[i].fy = 0.0;
            particles[i].fz = 0.0;
        }

        // Gravity — parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < actual_N; ++i) {
            particles[i].fx += particles[i].mass * cp.gx;
            particles[i].fy += particles[i].mass * cp.gy;
            particles[i].fz += particles[i].mass * cp.gz;
        }

        // Particle contacts — parallel with thread-local force accumulation
        // This avoids race conditions on force arrays
        int nthreads_actual = 1;
        #ifdef _OPENMP
        nthreads_actual = omp_get_max_threads();
        #endif
        std::vector<std::vector<double>> fx_local(nthreads_actual, std::vector<double>(actual_N, 0.0));
        std::vector<std::vector<double>> fy_local(nthreads_actual, std::vector<double>(actual_N, 0.0));
        std::vector<std::vector<double>> fz_local(nthreads_actual, std::vector<double>(actual_N, 0.0));

        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < actual_N - 1; ++i) {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            for (int j = i + 1; j < actual_N; ++j) {
                double rx = particles[j].x - particles[i].x;
                double ry = particles[j].y - particles[i].y;
                double rz = particles[j].z - particles[i].z;
                double dist2 = rx*rx + ry*ry + rz*rz;
                double rsum  = particles[i].radius + particles[j].radius;
                if (dist2 >= rsum*rsum) continue;

                double dist  = std::sqrt(dist2);
                double delta = rsum - dist;
                double nx = rx/dist, ny = ry/dist, nz = rz/dist;
                double dvx = particles[j].vx - particles[i].vx;
                double dvy = particles[j].vy - particles[i].vy;
                double dvz = particles[j].vz - particles[i].vz;
                double vn  = dvx*nx + dvy*ny + dvz*nz;
                double Fn  = std::max(0.0, cp.kn*delta - cp.gamma*vn);

                fx_local[tid][i] += Fn*nx;  fx_local[tid][j] -= Fn*nx;
                fy_local[tid][i] += Fn*ny;  fy_local[tid][j] -= Fn*ny;
                fz_local[tid][i] += Fn*nz;  fz_local[tid][j] -= Fn*nz;
            }
        }

        // Reduce thread-local forces
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < actual_N; ++i)
            for (int tid = 0; tid < nthreads_actual; ++tid) {
                particles[i].fx += fx_local[tid][i];
                particles[i].fy += fy_local[tid][i];
                particles[i].fz += fz_local[tid][i];
            }

        // Wall contacts — parallel (no race: each particle independent)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < actual_N; ++i) {
            double R_ = particles[i].radius;
            // Floor
            double d = R_ - particles[i].z;
            if (d > 0) particles[i].fz += std::max(0.0, cp.kn*d - cp.gamma*(-particles[i].vz));
            // Ceiling
            d = particles[i].z + R_ - cp.Lz;
            if (d > 0) particles[i].fz -= std::max(0.0, cp.kn*d - cp.gamma*(particles[i].vz));
            // X walls
            d = R_ - particles[i].x;
            if (d > 0) particles[i].fx += std::max(0.0, cp.kn*d - cp.gamma*(-particles[i].vx));
            d = particles[i].x + R_ - cp.Lx;
            if (d > 0) particles[i].fx -= std::max(0.0, cp.kn*d - cp.gamma*(particles[i].vx));
            // Y walls
            d = R_ - particles[i].y;
            if (d > 0) particles[i].fy += std::max(0.0, cp.kn*d - cp.gamma*(-particles[i].vy));
            d = particles[i].y + R_ - cp.Ly;
            if (d > 0) particles[i].fy -= std::max(0.0, cp.kn*d - cp.gamma*(particles[i].vy));
        }

        // Integrate — parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < actual_N; ++i) {
            particles[i].vx += (particles[i].fx / particles[i].mass) * dt;
            particles[i].vy += (particles[i].fy / particles[i].mass) * dt;
            particles[i].vz += (particles[i].fz / particles[i].mass) * dt;
            particles[i].x  += particles[i].vx * dt;
            particles[i].y  += particles[i].vy * dt;
            particles[i].z  += particles[i].vz * dt;
        }

        t += dt; ++step;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "  Total time: " << total << " s\n";

    // Append to scaling results file
    std::ofstream sf("scaling_" + tag + ".csv", std::ios::app);
    sf << nthreads << "," << total << "\n";
}

// ============================================================
//  NEIGHBOUR SEARCH SIMULATION
// ============================================================
void run_simulation_neighbour(int N, double tmax, const std::string& tag)
{
    std::cout << "\n=== Neighbour Search Simulation N=" << N << " ===\n";

    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    const double R  = 0.03, mass = 1.0;
    const double kn = 1e5, gamma = 50.0;
    const double g  = 9.81;
    const double dt = 1e-5;

    ContactParams cp{kn, gamma, 0.0, 0.0, -g, Lx, Ly, Lz};

    std::vector<Particle> particles;
    initialize_particles(particles, N, Lx, Ly, Lz, R, mass);

    NeighbourGrid grid;

    std::ofstream ke_file("ke_neighbour_" + tag + ".csv");
    ke_file << "t,KE\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    double t = 0.0;
    int step = 0;
    int out_interval = (int)(0.01 / dt);

    while (t <= tmax) {
        if (step % out_interval == 0) {
            double ke = compute_kinetic_energy(particles);
            ke_file << std::fixed << std::setprecision(6) << t << "," << ke << "\n";
        }

        zero_forces(particles);
        add_gravity(particles, cp);

        // Rebuild neighbour grid each step
        grid.build(particles, Lx, Ly, Lz, R);
        grid.compute_contacts(particles, cp);

        compute_wall_contacts(particles, cp);
        integrate(particles, dt);

        t += dt; ++step;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "  Total time (neighbour): " << total << " s\n";

    std::ofstream tf("timing_neighbour_" + tag + ".csv");
    tf << "N,time_s\n";
    tf << N << "," << total << "\n";
}

// ============================================================
//  MAIN
// ============================================================
int main(int argc, char* argv[])
{
    std::string mode = "all";
    if (argc > 1) mode = argv[1];

    if (mode == "all" || mode == "tests") {
        test_freefall();
        test_constant_velocity();
        test_bounce();
        test_timestep_convergence();
    }

    if (mode == "all" || mode == "serial") {
        run_simulation_serial(200,  0.5, "N200");
        run_simulation_serial(1000, 0.5, "N1000");
        run_simulation_serial(5000, 0.1, "N5000");
    }

    if (mode == "all" || mode == "parallel") {
        // Write header for scaling file
        for (const std::string& tag : {"N200", "N1000", "N5000"}) {
            std::ofstream sf("scaling_" + tag + ".csv");
            sf << "threads,time_s\n";
        }
        for (int threads : {1, 2, 4, 8}) {
            run_simulation_parallel(200,  0.5, threads, "N200");
            run_simulation_parallel(1000, 0.5, threads, "N1000");
            run_simulation_parallel(5000, 0.1, threads, "N5000");
        }
    }

    if (mode == "all" || mode == "neighbour") {
        run_simulation_neighbour(200,  0.5, "N200");
        run_simulation_neighbour(1000, 0.5, "N1000");
        run_simulation_neighbour(5000, 0.1, "N5000");
    }

    std::cout << "\nDone!\n";
    return 0;
}
