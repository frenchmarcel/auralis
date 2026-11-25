// Wave-band grid builder, bit for bit with the Python reference: build with -ffp-contract=off.
#include "aur_native.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// fdtd._dot3: fixed summation order, as the reference (never BLAS)
inline double dot3(const double* a, const double* b) {
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

inline double np_min(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return a < b ? a : b;
}

inline double np_clip01(double x) {
    if (x != x) return x;
    double y = x < 0.0 ? 0.0 : x;
    return y > 1.0 ? 1.0 : y;
}

double seg_dist(const double* P, const double* p, const double* q) {
    double pq[3], pp[3], r[3];
    for (int k = 0; k < 3; ++k) { pq[k] = q[k] - p[k]; pp[k] = P[k] - p[k]; }
    double t = np_clip01(dot3(pp, pq) / dot3(pq, pq));
    for (int k = 0; k < 3; ++k) r[k] = P[k] - (p[k] + t * pq[k]);
    return std::sqrt(dot3(r, r));
}

/** fdtd._point_tri_dist for one point. */
double point_tri_dist(const double* P, const double* a, const double* b, const double* c) {
    double ab[3], ac[3], ap[3], bp[3], cp[3];
    for (int k = 0; k < 3; ++k) {
        ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k];
        ap[k] = P[k] - a[k]; bp[k] = P[k] - b[k]; cp[k] = P[k] - c[k];
    }
    double d1 = dot3(ap, ab), d2 = dot3(ap, ac), d3 = dot3(bp, ab);
    double d4 = dot3(bp, ac), d5 = dot3(cp, ab), d6 = dot3(cp, ac);
    double va = d3 * d6 - d5 * d4;
    double vb = d5 * d2 - d1 * d6;
    double vc = d1 * d4 - d3 * d2;
    double denom = (va + vb) + vc;
    if (std::fabs(denom) < 1e-30) denom = 1e-30;
    double v = vb / denom, w = vc / denom;
    if (va >= 0 && vb >= 0 && vc >= 0) {
        double r[3];
        for (int k = 0; k < 3; ++k) r[k] = P[k] - ((a[k] + v * ab[k]) + w * ac[k]);
        return std::sqrt(dot3(r, r));
    }
    return np_min(np_min(seg_dist(P, a, b), seg_dist(P, b, c)), seg_dist(P, c, a));
}

inline void cross3(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

int resolve_threads(int32_t n_threads) {
    if (n_threads > 0) return n_threads;
    unsigned h = std::thread::hardware_concurrency();
    return h ? static_cast<int>(h) : 4;
}

/** Runs fn(chunk) for chunk in [0, n_chunks) on a pool, dynamic scheduling. */
template <class Fn>
void parallel_chunks(int64_t n_chunks, int n_threads, Fn fn) {
    std::atomic<int64_t> next(0);
    auto work = [&]() {
        for (;;) {
            int64_t c = next.fetch_add(1);
            if (c >= n_chunks) break;
            fn(c);
        }
    };
    int nt = static_cast<int>(std::min<int64_t>(n_threads, n_chunks));
    if (nt <= 1) { work(); return; }
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) pool.emplace_back(work);
    for (auto& th : pool) th.join();
}

struct TriBox { int64_t i0[3], i1[3]; bool ok; };

/** Box [i0, i1) of the cells (centres origin + i dx) within rad of the triangle bbox. */
TriBox tri_box(const double* a, const double* b, const double* c, const double* origin,
               double d, double rad, const int64_t* n) {
    TriBox t;
    t.ok = true;
    for (int k = 0; k < 3; ++k) {
        double lo = std::min(std::min(a[k], b[k]), c[k]) - rad;
        double hi = std::max(std::max(a[k], b[k]), c[k]) + rad;
        t.i0[k] = std::max<int64_t>(static_cast<int64_t>(std::floor((lo - origin[k]) / d)), 0);
        t.i1[k] = std::min<int64_t>(static_cast<int64_t>(std::ceil((hi - origin[k]) / d)) + 1, n[k]);
        if (t.i0[k] >= t.i1[k]) t.ok = false;
    }
    return t;
}

/** Triangles of each x chunk, in ascending index order (the per-cell update order). */
std::vector<std::vector<int32_t>> bucket_by_x(const std::vector<TriBox>& boxes, int64_t chunk,
                                              int64_t n_chunks) {
    std::vector<std::vector<int32_t>> out(n_chunks);
    for (size_t t = 0; t < boxes.size(); ++t) {
        if (!boxes[t].ok) continue;
        int64_t c0 = boxes[t].i0[0] / chunk, c1 = (boxes[t].i1[0] - 1) / chunk;
        for (int64_t c = c0; c <= c1; ++c) out[c].push_back(static_cast<int32_t>(t));
    }
    return out;
}

const int OFFS6[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

}  // namespace

extern "C" {

/** fdtd.voxelize, triangle loop: solid / matid / nrmq from the triangle soup (tv: n_tri x 9). */
AUR_API int32_t aur_grid_voxelize(const double* tv, int32_t n_tri, const int32_t* tri_mat,
                                  const double* origin, double dx, const int32_t* n32,
                                  uint8_t* solid, int32_t* matid, uint8_t* nrmq,
                                  int32_t n_threads) {
    const int64_t n[3] = {n32[0], n32[1], n32[2]};
    const int64_t n_cells = n[0] * n[1] * n[2];
    std::memset(solid, 0, static_cast<size_t>(n_cells));
    std::fill(matid, matid + n_cells, -1);
    std::memset(nrmq, 255, static_cast<size_t>(3 * n_cells));
    std::vector<float> best(static_cast<size_t>(n_cells), std::numeric_limits<float>::infinity());
    const double rad = (0.5 * std::sqrt(3.0)) * dx;
    std::vector<uint8_t> tnq(static_cast<size_t>(3) * n_tri);
    std::vector<TriBox> boxes(n_tri);
    for (int32_t t = 0; t < n_tri; ++t) {
        const double* a = tv + 9 * t; const double* b = a + 3; const double* c = a + 6;
        double e1[3], e2[3], tn[3];
        for (int k = 0; k < 3; ++k) { e1[k] = b[k] - a[k]; e2[k] = c[k] - a[k]; }
        cross3(e1, e2, tn);
        double nl = std::max(std::sqrt(dot3(tn, tn)), 1e-30);
        for (int k = 0; k < 3; ++k) {
            double q = std::min(std::fabs(tn[k] / nl) * 255.0, 255.0);
            tnq[3 * t + k] = static_cast<uint8_t>(q);
        }
        boxes[t] = tri_box(a, b, c, origin, dx, rad, n);
    }
    const int nt = resolve_threads(n_threads);
    const int64_t chunk = std::max<int64_t>(1, n[0] / (8 * nt));
    const int64_t n_chunks = (n[0] + chunk - 1) / chunk;
    auto buckets = bucket_by_x(boxes, chunk, n_chunks);
    parallel_chunks(n_chunks, nt, [&](int64_t ch) {
        const int64_t x_lo = ch * chunk, x_hi = std::min(n[0], x_lo + chunk);
        for (int32_t t : buckets[ch]) {
            const TriBox& bx = boxes[t];
            const double* a = tv + 9 * t; const double* b = a + 3; const double* c = a + 6;
            for (int64_t i = std::max(bx.i0[0], x_lo); i < std::min(bx.i1[0], x_hi); ++i)
                for (int64_t j = bx.i0[1]; j < bx.i1[1]; ++j)
                    for (int64_t k = bx.i0[2]; k < bx.i1[2]; ++k) {
                        double P[3] = {origin[0] + dx * static_cast<double>(i),
                                       origin[1] + dx * static_cast<double>(j),
                                       origin[2] + dx * static_cast<double>(k)};
                        double d = point_tri_dist(P, a, b, c);
                        if (!(d <= rad)) continue;
                        const int64_t id = (i * n[1] + j) * n[2] + k;
                        solid[id] = 1;
                        if (d < static_cast<double>(best[id])) {
                            matid[id] = tri_mat[t];
                            nrmq[3 * id] = tnq[3 * t];
                            nrmq[3 * id + 1] = tnq[3 * t + 1];
                            nrmq[3 * id + 2] = tnq[3 * t + 2];
                            best[id] = static_cast<float>(d);
                        }
                    }
        }
    });
    return 0;
}

/** fdtd._flood_air: the 6-connected non-solid component of the seed cell; returns its size. */
AUR_API int64_t aur_grid_flood(const uint8_t* solid, const int32_t* n32, int64_t seed,
                               uint8_t* air) {
    const int64_t n[3] = {n32[0], n32[1], n32[2]};
    const int64_t n_cells = n[0] * n[1] * n[2];
    std::memset(air, 0, static_cast<size_t>(n_cells));
    if (seed < 0 || seed >= n_cells || solid[seed]) return 0;
    std::vector<int64_t> cur{seed}, nxt;
    air[seed] = 1;
    int64_t count = 1;
    const int64_t sx = n[1] * n[2], sy = n[2];
    while (!cur.empty()) {
        nxt.clear();
        for (int64_t id : cur) {
            const int64_t i = id / sx, j = (id / sy) % n[1], k = id % n[2];
            const int64_t nb[6] = {i + 1 < n[0] ? id + sx : -1, i > 0 ? id - sx : -1,
                                   j + 1 < n[1] ? id + sy : -1, j > 0 ? id - sy : -1,
                                   k + 1 < n[2] ? id + 1 : -1, k > 0 ? id - 1 : -1};
            for (int64_t q : nb) {
                if (q < 0 || air[q] || solid[q]) continue;
                air[q] = 1;
                ++count;
                nxt.push_back(q);
            }
        }
        std::swap(cur, nxt);
    }
    return count;
}

/** conformal.fine_air_fractions after its RAM check: fine conservative raster at dx_f (centres
 *  org_f + i dx_f), fine air = open subcells of coarse air cells grown by at most n_iter
 *  6-connected steps through open subcells, frac = air subcells / F^3 per coarse cell. */
AUR_API int32_t aur_grid_fine_fraction(const double* tv, int32_t n_tri, const double* org_f,
                                       double dx_f, const int32_t* nc32, int32_t F,
                                       const uint8_t* air_coarse, int32_t n_iter, double* frac,
                                       int32_t n_threads) {
    const int64_t nc[3] = {nc32[0], nc32[1], nc32[2]};
    const int64_t n[3] = {nc[0] * F, nc[1] * F, nc[2] * F};
    const int64_t n_fine = n[0] * n[1] * n[2];
    // 0 = open, 1 = fine solid, 2 = fine air
    std::vector<uint8_t> mark(static_cast<size_t>(n_fine), 0);
    const double rad = (0.5 * std::sqrt(3.0)) * dx_f;
    std::vector<TriBox> boxes(n_tri);
    std::vector<double> nrms(static_cast<size_t>(3) * n_tri);
    for (int32_t t = 0; t < n_tri; ++t) {
        const double* a = tv + 9 * t; const double* b = a + 3; const double* c = a + 6;
        double e1[3], e2[3], nr[3];
        for (int k = 0; k < 3; ++k) { e1[k] = b[k] - a[k]; e2[k] = c[k] - a[k]; }
        cross3(e1, e2, nr);
        double nl = std::sqrt(dot3(nr, nr));
        if (nl < 1e-14) { boxes[t].ok = false; continue; }
        for (int k = 0; k < 3; ++k) nrms[3 * t + k] = nr[k] / nl;
        boxes[t] = tri_box(a, b, c, org_f, dx_f, rad, n);
    }
    const int nt = resolve_threads(n_threads);
    const int64_t chunk = std::max<int64_t>(1, n[0] / (8 * nt));
    const int64_t n_chunks = (n[0] + chunk - 1) / chunk;
    auto buckets = bucket_by_x(boxes, chunk, n_chunks);
    parallel_chunks(n_chunks, nt, [&](int64_t ch) {
        const int64_t x_lo = ch * chunk, x_hi = std::min(n[0], x_lo + chunk);
        for (int32_t t : buckets[ch]) {
            const TriBox& bx = boxes[t];
            const double* a = tv + 9 * t; const double* b = a + 3; const double* c = a + 6;
            const double* nr = &nrms[3 * t];
            for (int64_t i = std::max(bx.i0[0], x_lo); i < std::min(bx.i1[0], x_hi); ++i)
                for (int64_t j = bx.i0[1]; j < bx.i1[1]; ++j)
                    for (int64_t k = bx.i0[2]; k < bx.i1[2]; ++k) {
                        double P[3] = {org_f[0] + dx_f * static_cast<double>(i),
                                       org_f[1] + dx_f * static_cast<double>(j),
                                       org_f[2] + dx_f * static_cast<double>(k)};
                        double pa[3] = {P[0] - a[0], P[1] - a[1], P[2] - a[2]};
                        if (!(std::fabs(dot3(pa, nr)) <= rad)) continue;
                        if (point_tri_dist(P, a, b, c) <= rad)
                            mark[(i * n[1] + j) * n[2] + k] = 1;
                    }
        }
    });
    const int64_t sx = n[1] * n[2], sy = n[2];
    parallel_chunks(n[0], nt, [&](int64_t i) {
        for (int64_t j = 0; j < n[1]; ++j)
            for (int64_t k = 0; k < n[2]; ++k) {
                const int64_t id = i * sx + j * sy + k;
                if (mark[id] == 0 && air_coarse[((i / F) * nc[1] + j / F) * nc[2] + k / F])
                    mark[id] = 2;
            }
    });
    auto neighbours = [&](int64_t id, int64_t* nb) {
        const int64_t i = id / sx, j = (id / sy) % n[1], k = id % n[2];
        nb[0] = i + 1 < n[0] ? id + sx : -1; nb[1] = i > 0 ? id - sx : -1;
        nb[2] = j + 1 < n[1] ? id + sy : -1; nb[3] = j > 0 ? id - sy : -1;
        nb[4] = k + 1 < n[2] ? id + 1 : -1; nb[5] = k > 0 ? id - 1 : -1;
    };
    // the grown set does not depend on the order of the work, so both passes are threaded;
    // a cell is claimed once by an atomic exchange
    std::vector<std::vector<int64_t>> part(static_cast<size_t>(n[0]));
    parallel_chunks(n[0], nt, [&](int64_t i) {
        std::vector<int64_t>& v = part[static_cast<size_t>(i)];
        for (int64_t id = i * sx; id < (i + 1) * sx; ++id) {
            if (mark[id] != 2) continue;
            int64_t nb[6];
            neighbours(id, nb);
            for (int64_t q : nb)
                if (q >= 0 && mark[q] == 0) { v.push_back(id); break; }
        }
    });
    std::vector<int64_t> cur;
    for (auto& v : part) { cur.insert(cur.end(), v.begin(), v.end()); std::vector<int64_t>().swap(v); }
    for (int32_t it = 0; it < n_iter && !cur.empty(); ++it) {
        const int64_t m = static_cast<int64_t>(cur.size());
        const int64_t n_par = std::max<int64_t>(1, std::min<int64_t>(8 * nt, m / 4096 + 1));
        std::vector<std::vector<int64_t>> nx(static_cast<size_t>(n_par));
        parallel_chunks(n_par, nt, [&](int64_t c) {
            std::vector<int64_t>& v = nx[static_cast<size_t>(c)];
            for (int64_t e = c * m / n_par; e < (c + 1) * m / n_par; ++e) {
                int64_t nb[6];
                neighbours(cur[static_cast<size_t>(e)], nb);
                for (int64_t q : nb)
                    if (q >= 0 && mark[q] == 0
                        && __atomic_exchange_n(&mark[q], static_cast<uint8_t>(2), __ATOMIC_RELAXED) == 0)
                        v.push_back(q);
            }
        });
        cur.clear();
        for (auto& v : nx) cur.insert(cur.end(), v.begin(), v.end());
    }
    const double vol = static_cast<double>(F) * static_cast<double>(F) * static_cast<double>(F);
    parallel_chunks(nc[0], nt, [&](int64_t I) {
        for (int64_t J = 0; J < nc[1]; ++J)
            for (int64_t K = 0; K < nc[2]; ++K) {
                int64_t cnt = 0;
                for (int64_t a = 0; a < F; ++a)
                    for (int64_t b = 0; b < F; ++b) {
                        const int64_t base = ((I * F + a) * n[1] + (J * F + b)) * n[2] + K * F;
                        for (int64_t c = 0; c < F; ++c) cnt += mark[base + c] == 2;
                    }
                frac[(I * nc[1] + J) * nc[2] + K] = static_cast<double>(cnt) / vol;
            }
    });
    return F;
}

}  // extern "C"

namespace {

struct Poly {
    double v[24][3];
    int n = 0;
};

/** conformal._clip_poly_f (Sutherland-Hodgman, half-open convention). */
void clip_poly(const Poly& in, int axis, double value, bool keep_below, Poly& out) {
    out.n = 0;
    for (int i = 0; i < in.n; ++i) {
        const double* a = in.v[i];
        const double* b = in.v[(i + 1) % in.n];
        double da = a[axis] - value, db = b[axis] - value;
        bool ina = keep_below ? (da <= 0) : (da >= 0);
        bool inb = keep_below ? (db <= 0) : (db >= 0);
        if (ina) { std::memcpy(out.v[out.n], a, sizeof(double) * 3); ++out.n; }
        if (ina != inb) {
            double t = da / (da - db);
            for (int k = 0; k < 3; ++k) out.v[out.n][k] = a[k] + t * (b[k] - a[k]);
            ++out.n;
        }
    }
}

/** conformal._poly_area_f. */
double poly_area(const Poly& p) {
    if (p.n < 3) return 0.0;
    const double* p0 = p.v[0];
    double s[3] = {0.0, 0.0, 0.0};
    for (int i = 1; i < p.n - 1; ++i) {
        double u0 = p.v[i][0] - p0[0], u1 = p.v[i][1] - p0[1], u2 = p.v[i][2] - p0[2];
        double v0 = p.v[i + 1][0] - p0[0], v1 = p.v[i + 1][1] - p0[1], v2 = p.v[i + 1][2] - p0[2];
        s[0] += u1 * v2 - u2 * v1;
        s[1] += u2 * v0 - u0 * v2;
        s[2] += u0 * v1 - u1 * v0;
    }
    return 0.5 * std::sqrt(dot3(s, s));
}

struct Piece { int64_t cell; int32_t ax, side; double val; };

/** conformal._nearest_air_face: true and (cell, face) of the nearest air cell, or false.
 *  The BFS stays within max_depth <= 4 cells of the start: seen = a local 9^3 bitmap. */
bool nearest_air_face(const uint8_t* air, const int64_t* n, int64_t i, int64_t j, int64_t k,
                      int ax, int side, int max_depth, int64_t* cell_out, int* f_out) {
    constexpr int R = 4, S = 2 * R + 1;
    if (max_depth > R) max_depth = R;
    uint8_t seen[S * S * S];
    std::memset(seen, 0, sizeof(seen));
    auto local = [&](const std::array<int64_t, 3>& p) {
        return ((p[0] - i + R) * S + (p[1] - j + R)) * S + (p[2] - k + R);
    };
    std::vector<std::array<int64_t, 3>> queue{{i, j, k}}, nxt;
    seen[local(queue[0])] = 1;
    auto is_air = [&](int64_t a, int64_t b, int64_t c) { return air[(a * n[1] + b) * n[2] + c] != 0; };
    for (int depth = 0; depth < max_depth; ++depth) {
        nxt.clear();
        for (const auto& cell : queue) {
            for (const auto& o : OFFS6) {
                std::array<int64_t, 3> p{cell[0] + o[0], cell[1] + o[1], cell[2] + o[2]};
                if (!(0 <= p[0] && p[0] < n[0] && 0 <= p[1] && p[1] < n[1] && 0 <= p[2] && p[2] < n[2]))
                    continue;
                const int64_t lp = local(p);
                if (seen[lp]) continue;
                seen[lp] = 1;
                if (!is_air(p[0], p[1], p[2])) { nxt.push_back(p); continue; }
                int prefs[6];
                prefs[0] = 2 * ax + (side < 0 ? 1 : 0);
                int m = 1;
                for (int f = 0; f < 6; ++f) if (f != prefs[0]) prefs[m++] = f;
                for (int f : prefs) {
                    int a2 = f / 2;
                    int64_t sgn = (f % 2) ? 1 : -1;
                    std::array<int64_t, 3> q = p;
                    q[a2] += sgn;
                    bool inb = 0 <= q[a2] && q[a2] < n[a2];
                    if (!inb || !is_air(q[0], q[1], q[2])) {
                        *cell_out = (p[0] * n[1] + p[1]) * n[2] + p[2];
                        *f_out = f;
                        return true;
                    }
                }
            }
        }
        std::swap(queue, nxt);
        if (queue.empty()) break;
    }
    return false;
}

}  // namespace

extern "C" {

/** conformal.wall_face_areas: exact triangle pieces per solid cell (face_projections) deposited
 *  on the air faces facing them. W (n_air x 6) and cell_area (cells) must be zeroed; stats =
 *  (area_true, mapped, dropped, membrane, edge). Returns the number of (cell, axis, side) pieces. */
AUR_API int64_t aur_grid_wall_faces(const double* tv_in, int32_t n_tri, const double* origin_center,
                                    double dx, const int32_t* n32, const uint8_t* air,
                                    const int64_t* idx, int32_t max_walk, double* W,
                                    double* cell_area, double* stats, int32_t n_threads) {
    const int64_t n[3] = {n32[0], n32[1], n32[2]};
    const double eps = dx * 1e-9;
    const double shift[3] = {eps * 1.0, eps * 1.6180339887, eps * 2.6180339887};
    const double origin[3] = {origin_center[0] - 0.5 * dx, origin_center[1] - 0.5 * dx,
                              origin_center[2] - 0.5 * dx};
    const double rad = (0.5 * std::sqrt(3.0)) * dx;
    std::vector<std::vector<Piece>> pieces(n_tri);
    const int nt = resolve_threads(n_threads);
    const int64_t per = std::max<int64_t>(1, n_tri / (16 * nt));
    parallel_chunks((n_tri + per - 1) / per, nt, [&](int64_t ch) {
        for (int64_t t = ch * per; t < std::min<int64_t>(n_tri, (ch + 1) * per); ++t) {
            double tri[3][3];
            for (int q = 0; q < 3; ++q)
                for (int k = 0; k < 3; ++k) tri[q][k] = tv_in[9 * t + 3 * q + k] + shift[k];
            double e1[3], e2[3], nrm[3];
            for (int k = 0; k < 3; ++k) { e1[k] = tri[1][k] - tri[0][k]; e2[k] = tri[2][k] - tri[0][k]; }
            cross3(e1, e2, nrm);
            double a2 = std::sqrt(dot3(nrm, nrm));
            if (a2 < 1e-14) continue;
            for (int k = 0; k < 3; ++k) nrm[k] = nrm[k] / a2;
            int64_t lo[3], hi[3];
            bool ok = true;
            for (int k = 0; k < 3; ++k) {
                double mn = std::min(std::min(tri[0][k], tri[1][k]), tri[2][k]);
                double mx = std::max(std::max(tri[0][k], tri[1][k]), tri[2][k]);
                lo[k] = std::max<int64_t>(static_cast<int64_t>(std::floor((mn - origin[k]) / dx)) - 1, 0);
                hi[k] = std::min<int64_t>(static_cast<int64_t>(std::floor((mx - origin[k]) / dx)) + 1, n[k] - 1);
                if (lo[k] > hi[k]) ok = false;
            }
            if (!ok) continue;
            double w_ax[3];
            int sides[3];
            for (int k = 0; k < 3; ++k) { w_ax[k] = nrm[k] * nrm[k]; sides[k] = nrm[k] > 0 ? 1 : -1; }
            std::vector<Piece>& out = pieces[t];
            Poly p0, p1, p2;
            for (int64_t i = lo[0]; i <= hi[0]; ++i)
                for (int64_t j = lo[1]; j <= hi[1]; ++j)
                    for (int64_t k = lo[2]; k <= hi[2]; ++k) {
                        const int64_t id = (i * n[1] + j) * n[2] + k;
                        if (air[id]) continue;
                        double ctr[3] = {origin[0] + (static_cast<double>(i) + 0.5) * dx,
                                         origin[1] + (static_cast<double>(j) + 0.5) * dx,
                                         origin[2] + (static_cast<double>(k) + 0.5) * dx};
                        double r[3] = {ctr[0] - tri[0][0], ctr[1] - tri[0][1], ctr[2] - tri[0][2]};
                        if (!(std::fabs(dot3(r, nrm)) <= rad)) continue;
                        const double c0[3] = {origin[0] + static_cast<double>(i) * dx,
                                              origin[1] + static_cast<double>(j) * dx,
                                              origin[2] + static_cast<double>(k) * dx};
                        p0.n = 3;
                        for (int q = 0; q < 3; ++q) std::memcpy(p0.v[q], tri[q], sizeof(double) * 3);
                        bool keep = true;
                        for (int a = 0; a < 3 && keep; ++a) {
                            clip_poly(p0, a, c0[a], false, p1);
                            if (p1.n < 3) { keep = false; break; }
                            clip_poly(p1, a, c0[a] + dx, true, p2);
                            if (p2.n < 3) { keep = false; break; }
                            p0 = p2;
                        }
                        if (!keep) continue;
                        double A = poly_area(p0);
                        if (A <= 0) continue;
                        for (int a = 0; a < 3; ++a) {
                            if (w_ax[a] < 1e-6) continue;
                            out.push_back({id, a, sides[a], A * w_ax[a]});
                        }
                    }
        }
    });
    // dict accumulation in insertion order (triangle, cell, axis), as the Python reference
    std::unordered_map<int64_t, int64_t> slot;
    std::vector<int64_t> keys;
    std::vector<double> vals;
    for (int32_t t = 0; t < n_tri; ++t)
        for (const Piece& pc : pieces[t]) {
            const int64_t key = (pc.cell * 3 + pc.ax) * 2 + (pc.side > 0 ? 1 : 0);
            auto it = slot.find(key);
            if (it == slot.end()) {
                slot.emplace(key, static_cast<int64_t>(keys.size()));
                keys.push_back(key);
                vals.push_back(0.0 + pc.val);
            } else {
                vals[it->second] = vals[it->second] + pc.val;
            }
        }
    pieces.clear();
    double a_tot = 0.0, a_mapped = 0.0, a_dropped = 0.0, a_membrane = 0.0, a_edge = 0.0;
    for (size_t s = 0; s < keys.size(); ++s) {
        const int64_t key = keys[s];
        const int side = (key % 2) ? 1 : -1;
        const int ax = static_cast<int>((key / 2) % 3);
        const int64_t cell = key / 6;
        const int64_t ci[3] = {cell / (n[1] * n[2]), (cell / n[2]) % n[1], cell % n[2]};
        const double A = vals[s];
        a_tot += A;
        int found = 0;
        for (int sgn : {side, -side}) {
            int64_t p[3] = {ci[0], ci[1], ci[2]};
            for (int step = 0; step < max_walk; ++step) {
                p[ax] += sgn;
                if (!(0 <= p[ax] && p[ax] < n[ax])) break;
                const int64_t pid = (p[0] * n[1] + p[1]) * n[2] + p[2];
                if (air[pid]) {
                    const int f_air = 2 * ax + (sgn < 0 ? 1 : 0);
                    W[6 * idx[pid] + f_air] += A;
                    ++found;
                    break;
                }
            }
        }
        if (found == 0) {
            int64_t hc;
            int hf;
            if (nearest_air_face(air, n, ci[0], ci[1], ci[2], ax, side, 4, &hc, &hf)) {
                W[6 * idx[hc] + hf] += A;
                a_edge += A;
                found = 1;
            }
        }
        if (found == 0) {
            a_dropped += A;
        } else {
            a_mapped += A;
            if (found == 2) a_membrane += A;
        }
    }
    for (size_t s = 0; s < keys.size(); ++s) cell_area[keys[s] / 6] += vals[s];
    stats[0] = a_tot; stats[1] = a_mapped; stats[2] = a_dropped;
    stats[3] = a_membrane; stats[4] = a_edge;
    return static_cast<int64_t>(keys.size());
}

}  // extern "C"

extern "C" {

/** conformal_corrections, default path: volume donation of the solid cells to their air
 *  6-neighbours (deep donors relayed by the direct ones), then smooth_sweeps conservative
 *  diffusion sweeps. frac includes the slab; phi (n_air) = 1 + excess, before the cap;
 *  orphan = deep donors with no direct neighbour (their volume is lost). Shifts in the order
 *  of the Python reference: neighbours i-1, i+1, j-1, j+1, k-1, k+1. Returns 1 if donors. */
AUR_API int32_t aur_grid_donate(const int32_t* n32, const uint8_t* air, const double* frac,
                                int32_t smooth_sweeps, double t_diff, double* phi,
                                uint8_t* orphan, int32_t n_threads) {
    const int64_t n[3] = {n32[0], n32[1], n32[2]};
    const int64_t N = n[0] * n[1] * n[2];
    const int64_t sx = n[1] * n[2], sy = n[2];
    const int nt = resolve_threads(n_threads);
    auto nbs = [&](int64_t id, int64_t* nb) {
        const int64_t i = id / sx, j = (id / sy) % n[1], k = id % n[2];
        nb[0] = i > 0 ? id - sx : -1; nb[1] = i + 1 < n[0] ? id + sx : -1;
        nb[2] = j > 0 ? id - sy : -1; nb[3] = j + 1 < n[1] ? id + sy : -1;
        nb[4] = k > 0 ? id - 1 : -1;  nb[5] = k + 1 < n[2] ? id + 1 : -1;
    };
    auto slabs = [&](auto fn) {
        parallel_chunks(n[0], nt, [&](int64_t i) {
            for (int64_t id = i * sx; id < (i + 1) * sx; ++id) fn(id);
        });
    };
    std::memset(orphan, 0, static_cast<size_t>(N));
    std::vector<int8_t> cnt(static_cast<size_t>(N));
    std::vector<uint8_t> flag(static_cast<size_t>(N));        // 1 = direct, 2 = deep donor
    std::atomic<int> any_donor(0), any_deep(0);
    slabs([&](int64_t id) {
        int64_t nb[6];
        nbs(id, nb);
        int c = 0;
        for (int64_t q : nb) c += (q >= 0 && air[q]) ? 1 : 0;
        cnt[id] = static_cast<int8_t>(c);
        uint8_t f = 0;
        if (!air[id] && frac[id] > 1e-9) {
            f = c > 0 ? 1 : 2;
            any_donor.store(1, std::memory_order_relaxed);
            if (f == 2) any_deep.store(1, std::memory_order_relaxed);
        }
        flag[id] = f;
    });
    std::vector<double> ex(static_cast<size_t>(N), 0.0);
    if (any_donor.load()) {
        std::vector<double> give(static_cast<size_t>(N), 0.0), share(static_cast<size_t>(N), 0.0);
        if (any_deep.load()) {
            slabs([&](int64_t id) {
                if (flag[id] != 2) return;
                int64_t nb[6];
                nbs(id, nb);
                int c = 0;
                for (int64_t q : nb) c += (q >= 0 && flag[q] == 1) ? 1 : 0;
                if (c > 0) give[id] = frac[id] / static_cast<double>(c);
                else orphan[id] = 1;
            });
        }
        slabs([&](int64_t id) {
            if (flag[id] != 1) return;
            int64_t nb[6];
            nbs(id, nb);
            double pool = frac[id];
            if (any_deep.load(std::memory_order_relaxed))
                for (int64_t q : nb) pool = pool + (q >= 0 ? give[q] : 0.0);
            share[id] = pool / static_cast<double>(std::max<int>(cnt[id], 1));
        });
        slabs([&](int64_t id) {
            if (!air[id]) return;
            int64_t nb[6];
            nbs(id, nb);
            double e = 0.0;
            for (int64_t q : nb) e = e + (q >= 0 ? share[q] : 0.0);
            ex[id] = e;
        });
        std::vector<double> ex2(static_cast<size_t>(N), 0.0);
        for (int32_t it = 0; it < smooth_sweeps; ++it) {
            slabs([&](int64_t id) {
                double delta = 0.0;
                if (air[id]) {
                    int64_t nb[6];
                    nbs(id, nb);
                    for (int64_t q : nb)
                        delta = delta + ((q >= 0 && air[q]) ? (ex[q] - ex[id]) : 0.0);
                }
                ex2[id] = ex[id] + t_diff * delta;
            });
            std::swap(ex, ex2);
        }
    }
    int64_t a = 0;
    for (int64_t id = 0; id < N; ++id)
        if (air[id]) phi[a++] = 1.0 + ex[id];
    return any_donor.load() ? 1 : 0;
}

}  // extern "C"

extern "C" {

/** FdtdGrid index and neighbours: idx (grid cells, -1 = not air), cells (n_air x 3, C order),
 *  nb (n_air x 6, faces -x +x -y +y -z +z, ghost = n_air), K = number of air neighbours; if
 *  nb2 is given, the second neighbours along each direction and full = 1 where all 12 are
 *  air. Returns n_air. */
AUR_API int64_t aur_grid_structure(const uint8_t* air, const int32_t* n32, int32_t* idx,
                                   int32_t* cells, int32_t* nb, double* K, int32_t* nb2,
                                   uint8_t* full, int32_t n_threads) {
    const int64_t n[3] = {n32[0], n32[1], n32[2]};
    const int64_t plane = n[1] * n[2];
    const int nt = resolve_threads(n_threads);
    std::vector<int64_t> base(static_cast<size_t>(n[0] + 1), 0);
    parallel_chunks(n[0], nt, [&](int64_t i) {
        int64_t c = 0;
        for (int64_t q = i * plane; q < (i + 1) * plane; ++q) c += air[q] ? 1 : 0;
        base[static_cast<size_t>(i + 1)] = c;
    });
    for (int64_t i = 0; i < n[0]; ++i) base[i + 1] += base[i];
    const int64_t n_air = base[n[0]];
    const int32_t ghost = static_cast<int32_t>(n_air);
    parallel_chunks(n[0], nt, [&](int64_t i) {
        int64_t a = base[i];
        for (int64_t j = 0; j < n[1]; ++j)
            for (int64_t k = 0; k < n[2]; ++k) {
                const int64_t q = i * plane + j * n[2] + k;
                if (air[q]) {
                    idx[q] = static_cast<int32_t>(a);
                    cells[3 * a] = static_cast<int32_t>(i);
                    cells[3 * a + 1] = static_cast<int32_t>(j);
                    cells[3 * a + 2] = static_cast<int32_t>(k);
                    ++a;
                } else {
                    idx[q] = -1;
                }
            }
    });
    parallel_chunks(n[0], nt, [&](int64_t i) {
        for (int64_t a = base[i]; a < base[i + 1]; ++a) {
            const int64_t c[3] = {cells[3 * a], cells[3 * a + 1], cells[3 * a + 2]};
            int cnt = 0;
            for (int f = 0; f < 6; ++f) {
                const int ax = f / 2;
                const int64_t s = (f % 2) ? 1 : -1;
                int64_t p[3] = {c[0], c[1], c[2]};
                p[ax] += s;
                int32_t v = ghost;
                if (p[ax] >= 0 && p[ax] < n[ax]) {
                    const int32_t w = idx[p[0] * plane + p[1] * n[2] + p[2]];
                    if (w >= 0) { v = w; ++cnt; }
                }
                nb[6 * a + f] = v;
            }
            K[a] = static_cast<double>(cnt);
        }
    });
    if (nb2) {
        parallel_chunks(n[0], nt, [&](int64_t i) {
            for (int64_t a = base[i]; a < base[i + 1]; ++a) {
                bool all = true;
                for (int f = 0; f < 6; ++f) {
                    const int32_t v = nb[6 * a + f];
                    const int32_t w = (v == ghost) ? ghost : nb[6 * static_cast<int64_t>(v) + f];
                    nb2[6 * a + f] = w;
                    all = all && v != ghost && w != ghost;
                }
                if (full) full[a] = all ? 1 : 0;
            }
        });
    }
    return n_air;
}

/** CPML cells of one axis: air cells less than npml - 1 cells from either face of that axis
 *  (u > 0), with their 6 neighbours all air, ascending; locm = position in that list of the
 *  minus neighbour along the axis, or -1. sel and locm hold n_air entries; returns the count. */
AUR_API int64_t aur_grid_pml_axis(const int32_t* cells, const int32_t* nb, int64_t n_air,
                                  const int32_t* n32, int32_t npml, int32_t axis, int64_t* sel,
                                  int32_t* locm, int32_t n_threads) {
    const int64_t shp = n32[axis];
    const int nt = resolve_threads(n_threads);
    const int64_t chunk = std::max<int64_t>(1 << 16, n_air / (8 * nt) + 1);
    const int64_t n_chunks = (n_air + chunk - 1) / chunk;
    auto picked = [&](int64_t a) {
        const int64_t ia = cells[3 * a + axis];
        if (!(ia < npml - 1 || ia > shp - npml)) return false;
        for (int f = 0; f < 6; ++f)
            if (nb[6 * a + f] >= n_air) return false;
        return true;
    };
    std::vector<int64_t> cnt(static_cast<size_t>(n_chunks + 1), 0);
    parallel_chunks(n_chunks, nt, [&](int64_t c) {
        int64_t m = 0;
        for (int64_t a = c * chunk; a < std::min(n_air, (c + 1) * chunk); ++a) m += picked(a);
        cnt[static_cast<size_t>(c + 1)] = m;
    });
    for (int64_t c = 0; c < n_chunks; ++c) cnt[c + 1] += cnt[c];
    parallel_chunks(n_chunks, nt, [&](int64_t c) {
        int64_t m = cnt[c];
        for (int64_t a = c * chunk; a < std::min(n_air, (c + 1) * chunk); ++a)
            if (picked(a)) sel[m++] = a;
    });
    const int64_t n_sel = cnt[n_chunks];
    parallel_chunks((n_sel + chunk - 1) / chunk, nt, [&](int64_t c) {
        for (int64_t k = c * chunk; k < std::min(n_sel, (c + 1) * chunk); ++k) {
            const int64_t nm = nb[6 * sel[k] + 2 * axis];
            const int64_t* hit = std::lower_bound(sel, sel + n_sel, nm);
            locm[k] = (hit != sel + n_sel && *hit == nm) ? static_cast<int32_t>(hit - sel) : -1;
        }
    });
    return n_sel;
}

/** 1 where an air cell is a CPML cell or reaches one through nb or nb2 (pmlm: n_air + 1
 *  entries, the ghost one 0): the cells kept at order 2 around the absorbing layer. */
AUR_API void aur_grid_collar(const int32_t* nb, const int32_t* nb2, int64_t n_air,
                             const uint8_t* pmlm, uint8_t* out, int32_t n_threads) {
    const int nt = resolve_threads(n_threads);
    const int64_t chunk = std::max<int64_t>(1 << 16, n_air / (8 * nt) + 1);
    parallel_chunks((n_air + chunk - 1) / chunk, nt, [&](int64_t c) {
        for (int64_t a = c * chunk; a < std::min(n_air, (c + 1) * chunk); ++a) {
            bool b = pmlm[a] != 0;
            for (int f = 0; f < 6 && !b; ++f)
                b = pmlm[nb[6 * a + f]] != 0 || pmlm[nb2[6 * a + f]] != 0;
            out[a] = b ? 1 : 0;
        }
    });
}

}  // extern "C"

extern "C" {

/** Wall faces of direction f (-x +x -y +y -z +z): air cells whose neighbour there is the
 *  ghost, ascending (wi, n_air entries reserved); inb = 1 if that neighbour is a solid cell
 *  inside the grid, 0 at the grid border. Returns their number. */
AUR_API int64_t aur_grid_wall_list(const int32_t* cells, const int32_t* nb, int64_t n_air,
                                   const int32_t* n32, int32_t f, int64_t* wi, uint8_t* inb,
                                   int32_t n_threads) {
    const int ax = f / 2;
    const int32_t sgn = (f % 2) ? 1 : -1;
    const int nt = resolve_threads(n_threads);
    const int64_t chunk = std::max<int64_t>(1 << 16, n_air / (8 * nt) + 1);
    const int64_t n_chunks = (n_air + chunk - 1) / chunk;
    std::vector<int64_t> cnt(static_cast<size_t>(n_chunks + 1), 0);
    parallel_chunks(n_chunks, nt, [&](int64_t c) {
        int64_t m = 0;
        for (int64_t a = c * chunk; a < std::min(n_air, (c + 1) * chunk); ++a)
            m += nb[6 * a + f] >= n_air;
        cnt[static_cast<size_t>(c + 1)] = m;
    });
    for (int64_t c = 0; c < n_chunks; ++c) cnt[c + 1] += cnt[c];
    parallel_chunks(n_chunks, nt, [&](int64_t c) {
        int64_t m = cnt[c];
        for (int64_t a = c * chunk; a < std::min(n_air, (c + 1) * chunk); ++a) {
            if (nb[6 * a + f] < n_air) continue;
            const int32_t q = cells[3 * a + ax] + sgn;
            wi[m] = a;
            inb[m] = (q >= 0 && q < n32[ax]) ? 1 : 0;
            ++m;
        }
    });
    return cnt[n_chunks];
}

}  // extern "C"

extern "C" {

/** FdtdGrid coefficients, numpy's elementwise operations in its order: Gamma = half_lam gam,
 *  den = (ph + Gamma) + s_half, c_center = 2 ph - K lam2, c_old = ((Gamma + s_half) - ph) / den,
 *  c_now = c_center / den, c_nb = lam2 / den, and c_drv = lam_dt / den when c_drv is given. */
AUR_API void aur_grid_coefficients(int64_t n_air, double half_lam, double lam2, double lam_dt,
                                   const double* gam, const double* K, const double* s_half,
                                   const double* ph, double* Gamma, double* c_center,
                                   double* c_old, double* c_now, double* c_nb, double* c_drv,
                                   int32_t n_threads) {
    const int nt = resolve_threads(n_threads);
    const int64_t chunk = std::max<int64_t>(1 << 16, n_air / (8 * nt) + 1);
    parallel_chunks((n_air + chunk - 1) / chunk, nt, [&](int64_t c) {
        for (int64_t i = c * chunk; i < std::min(n_air, (c + 1) * chunk); ++i) {
            const double g = half_lam * gam[i];
            const double den = (ph[i] + g) + s_half[i];
            const double cc = 2.0 * ph[i] - K[i] * lam2;
            Gamma[i] = g;
            c_center[i] = cc;
            c_old[i] = ((g + s_half[i]) - ph[i]) / den;
            c_now[i] = cc / den;
            c_nb[i] = lam2 / den;
            if (c_drv) c_drv[i] = lam_dt / den;
        }
    });
}

}  // extern "C"
