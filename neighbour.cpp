#include "neighbour.h"
#include <cmath>
#include <algorithm>

int NeighbourGrid::cell_index(int cx, int cy, int cz) const
{
    return cx + ncx * (cy + ncy * cz);
}

void NeighbourGrid::particle_cell(const Particle& pi, int& cx, int& cy, int& cz) const
{
    cx = std::min((int)(pi.x / cell_size), ncx - 1);
    cy = std::min((int)(pi.y / cell_size), ncy - 1);
    cz = std::min((int)(pi.z / cell_size), ncz - 1);
    cx = std::max(cx, 0);
    cy = std::max(cy, 0);
    cz = std::max(cz, 0);
}

void NeighbourGrid::build(const std::vector<Particle>& p,
                           double lx, double ly, double lz,
                           double max_radius)
{
    Lx = lx; Ly = ly; Lz = lz;
    cell_size = 2.0 * max_radius;  // minimum cell size = diameter

    ncx = std::max(1, (int)(Lx / cell_size));
    ncy = std::max(1, (int)(Ly / cell_size));
    ncz = std::max(1, (int)(Lz / cell_size));

    int total_cells = ncx * ncy * ncz;
    cell_head.assign(total_cells, -1);
    next.assign(p.size(), -1);

    for (int i = 0; i < (int)p.size(); ++i) {
        int cx, cy, cz;
        particle_cell(p[i], cx, cy, cz);
        int c = cell_index(cx, cy, cz);
        next[i] = cell_head[c];
        cell_head[c] = i;
    }
}

void NeighbourGrid::compute_contacts(std::vector<Particle>& p, const ContactParams& cp)
{
    int N = (int)p.size();

    for (int i = 0; i < N; ++i) {
        int cxi, cyi, czi;
        particle_cell(p[i], cxi, cyi, czi);

        // Search 3x3x3 neighbourhood of cells
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx_ = cxi + dx;
            int ny_ = cyi + dy;
            int nz_ = czi + dz;

            if (nx_ < 0 || nx_ >= ncx) continue;
            if (ny_ < 0 || ny_ >= ncy) continue;
            if (nz_ < 0 || nz_ >= ncz) continue;

            int c = cell_index(nx_, ny_, nz_);
            int j = cell_head[c];

            while (j != -1) {
                if (j > i) {  // only process each pair once (i < j)
                    double rx = p[j].x - p[i].x;
                    double ry = p[j].y - p[i].y;
                    double rz = p[j].z - p[i].z;
                    double dist2 = rx*rx + ry*ry + rz*rz;
                    double rsum  = p[i].radius + p[j].radius;

                    if (dist2 < rsum*rsum && dist2 > 1e-28) {
                        double dist  = std::sqrt(dist2);
                        double delta = rsum - dist;

                        double nx = rx / dist;
                        double ny = ry / dist;
                        double nz = rz / dist;

                        double dvx = p[j].vx - p[i].vx;
                        double dvy = p[j].vy - p[i].vy;
                        double dvz = p[j].vz - p[i].vz;
                        double vn  = dvx*nx + dvy*ny + dvz*nz;

                        double Fn = std::max(0.0, cp.kn*delta - cp.gamma*vn);

                        double fx = Fn * nx;
                        double fy = Fn * ny;
                        double fz = Fn * nz;

                        p[i].fx += fx;  p[j].fx -= fx;
                        p[i].fy += fy;  p[j].fy -= fy;
                        p[i].fz += fz;  p[j].fz -= fz;
                    }
                }
                j = next[j];
            }
        }
    }
}
