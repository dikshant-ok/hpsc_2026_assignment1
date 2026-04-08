#include "forces.h"
#include <cmath>
#include <algorithm>

void zero_forces(std::vector<Particle>& p)
{
    for (auto& pi : p) {
        pi.fx = 0.0;
        pi.fy = 0.0;
        pi.fz = 0.0;
    }
}

void add_gravity(std::vector<Particle>& p, const ContactParams& cp)
{
    for (auto& pi : p) {
        pi.fx += pi.mass * cp.gx;
        pi.fy += pi.mass * cp.gy;
        pi.fz += pi.mass * cp.gz;
    }
}

// Compute contact force between two overlapping particles
// Returns force ON particle i (reaction on j is equal and opposite)
static void contact_force(const Particle& pi, const Particle& pj,
                           double& fx, double& fy, double& fz,
                           double kn, double gamma)
{
    double rx = pj.x - pi.x;
    double ry = pj.y - pi.y;
    double rz = pj.z - pi.z;
    double dist = std::sqrt(rx*rx + ry*ry + rz*rz);
    if (dist < 1e-14) return;

    double delta = pi.radius + pj.radius - dist;
    if (delta <= 0.0) return;

    // Unit normal from i to j
    double nx = rx / dist;
    double ny = ry / dist;
    double nz = rz / dist;

    // Normal relative velocity (j relative to i, projected onto normal)
    double dvx = pj.vx - pi.vx;
    double dvy = pj.vy - pi.vy;
    double dvz = pj.vz - pi.vz;
    double vn = dvx*nx + dvy*ny + dvz*nz;

    // Spring-dashpot force magnitude (clamped to prevent attraction)
    double Fn = std::max(0.0, kn*delta - gamma*vn);

    fx = Fn * nx;
    fy = Fn * ny;
    fz = Fn * nz;
}

void compute_particle_contacts(std::vector<Particle>& p, const ContactParams& cp)
{
    int N = (int)p.size();

    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double rx = p[j].x - p[i].x;
            double ry = p[j].y - p[i].y;
            double rz = p[j].z - p[i].z;
            double dist2 = rx*rx + ry*ry + rz*rz;
            double rsum  = p[i].radius + p[j].radius;
            if (dist2 >= rsum*rsum) continue;  // early exit — no contact

            double fx=0, fy=0, fz=0;
            contact_force(p[i], p[j], fx, fy, fz, cp.kn, cp.gamma);

            // Newton's third law
            p[i].fx += fx;
            p[i].fy += fy;
            p[i].fz += fz;
            p[j].fx -= fx;
            p[j].fy -= fy;
            p[j].fz -= fz;
        }
    }
}

void compute_wall_contacts(std::vector<Particle>& p, const ContactParams& cp)
{
    for (auto& pi : p) {
        double R = pi.radius;
        double kn = cp.kn;
        double gm = cp.gamma;

        // --- Lower z wall (floor) ---
        {
            double delta = R - pi.z;
            if (delta > 0.0) {
                double vn = -pi.vz;  // velocity into wall (wall normal is +z)
                double Fn = std::max(0.0, kn*delta - gm*vn);
                pi.fz += Fn;
            }
        }
        // --- Upper z wall (ceiling) ---
        {
            double delta = pi.z + R - cp.Lz;
            if (delta > 0.0) {
                double vn = pi.vz;
                double Fn = std::max(0.0, kn*delta - gm*vn);
                pi.fz -= Fn;
            }
        }
        // --- Lower x wall ---
        {
            double delta = R - pi.x;
            if (delta > 0.0) {
                double vn = -pi.vx;
                double Fn = std::max(0.0, kn*delta - gm*vn);
                pi.fx += Fn;
            }
        }
        // --- Upper x wall ---
        {
            double delta = pi.x + R - cp.Lx;
            if (delta > 0.0) {
                double vn = pi.vx;
                double Fn = std::max(0.0, kn*delta - gm*vn);
                pi.fx -= Fn;
            }
        }
        // --- Lower y wall ---
        {
            double delta = R - pi.y;
            if (delta > 0.0) {
                double vn = -pi.vy;
                double Fn = std::max(0.0, kn*delta - gm*vn);
                pi.fy += Fn;
            }
        }
        // --- Upper y wall ---
        {
            double delta = pi.y + R - cp.Ly;
            if (delta > 0.0) {
                double vn = pi.vy;
                double Fn = std::max(0.0, kn*delta - gm*vn);
                pi.fy -= Fn;
            }
        }
    }
}
