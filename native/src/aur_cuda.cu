/* Auralis CUDA core (drop-in for libauralis); -DAUR_EMU builds the same code for the CPU. */
#include "aur_native.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#if defined(__CUDACC__)
  #define AUR_GPU 1
  #include <cuda_runtime.h>
  #define DEVFN __host__ __device__ __forceinline__
  #define DEVMETH __host__ __device__ __forceinline__
  #define KERNFN __device__ __forceinline__
  #define GLOBALFN __global__
#else
  #ifndef AUR_EMU
    #error "Build with nvcc, or with -DAUR_EMU for emulation mode."
  #endif
  #define AUR_GPU 0
  #define DEVFN static inline
  #define DEVMETH inline
  #define KERNFN static inline
  #define GLOBALFN static
#endif

/* Shared host + device math */
struct AV3 { float x, y, z; };
DEVFN AV3 av3(float x, float y, float z) { AV3 r; r.x = x; r.y = y; r.z = z; return r; }
DEVFN AV3 vadd(AV3 a, AV3 b) { return av3(a.x + b.x, a.y + b.y, a.z + b.z); }
DEVFN AV3 vsub(AV3 a, AV3 b) { return av3(a.x - b.x, a.y - b.y, a.z - b.z); }
DEVFN AV3 vscale(AV3 a, float s) { return av3(a.x * s, a.y * s, a.z * s); }
DEVFN float vdot(AV3 a, AV3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
DEVFN AV3 vcross(AV3 a, AV3 b) {
    return av3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x);
}
DEVFN float vlen(AV3 a) { return sqrtf(vdot(a, a)); }
DEVFN AV3 vnorm(AV3 a) {
    float l = vlen(a);
    return l > 0.f ? vscale(a, 1.f / l) : av3(0.f, 0.f, 1.f);
}

struct ABvhNode { AV3 lo; int32_t left; AV3 hi; int32_t count; };

/* PCG32 (same as the CPU) */
struct APcg32 {
    uint64_t state, inc;
    DEVMETH void init(uint64_t seed, uint64_t seq) {
        state = 0u; inc = (seq << 1u) | 1u;
        next(); state += seed; next();
    }
    DEVMETH uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xs >> rot) | (xs << ((-rot) & 31));
    }
    DEVMETH float f01() { return (next() >> 8) * (1.0f / 16777216.0f); }
};

/* Atomics: real on GPU, plain adds in emulation */
#if AUR_GPU
__device__ __forceinline__ void aurAtomicAddF(float* a, float v) { atomicAdd(a, v); }
__device__ __forceinline__ void aurAtomicAddD(double* a, double v) {
  #if __CUDA_ARCH__ >= 600
    atomicAdd(a, v);
  #else
    unsigned long long* p = (unsigned long long*)a;
    unsigned long long old = *p, assumed;
    do { assumed = old;
         old = atomicCAS(p, assumed,
               __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
  #endif
}
#else
DEVFN void aurAtomicAddF(float* a, float v) { *a += v; }
DEVFN void aurAtomicAddD(double* a, double v) { *a += v; }
#endif

/* Device scene: device pointers on GPU, host pointers in EMU */
struct DevScene {
    const ABvhNode* nodes;
    const int32_t*  right;
    const AV3* v0; const AV3* e1; const AV3* e2; const AV3* ng;
    const int32_t*  mat;
    int32_t n_tris;
};

#define AUR_RAY_EPS 1e-4f
#define AUR_STACK   96

DEVFN bool d_hit_aabb(const ABvhNode& n, AV3 o, AV3 inv_d, float t_max) {
    float tx1 = (n.lo.x - o.x) * inv_d.x, tx2 = (n.hi.x - o.x) * inv_d.x;
    float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);
    float ty1 = (n.lo.y - o.y) * inv_d.y, ty2 = (n.hi.y - o.y) * inv_d.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2)); tmax = fminf(tmax, fmaxf(ty1, ty2));
    float tz1 = (n.lo.z - o.z) * inv_d.z, tz2 = (n.hi.z - o.z) * inv_d.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2)); tmax = fminf(tmax, fmaxf(tz1, tz2));
    return tmax >= fmaxf(tmin, 0.0f) && tmin < t_max;
}

DEVFN bool d_hit_tri(const DevScene& s, int32_t i, AV3 o, AV3 d,
                     float t_max, float* t_out) {
    AV3 p = vcross(d, s.e2[i]);
    float det = vdot(s.e1[i], p);
    if (fabsf(det) < 1e-12f) return false;
    float inv = 1.f / det;
    AV3 tv = vsub(o, s.v0[i]);
    float u = vdot(tv, p) * inv;
    if (u < -1e-6f || u > 1.f + 1e-6f) return false;
    AV3 q = vcross(tv, s.e1[i]);
    float v = vdot(d, q) * inv;
    if (v < -1e-6f || u + v > 1.f + 1e-6f) return false;
    float t = vdot(s.e2[i], q) * inv;
    if (t <= AUR_RAY_EPS || t >= t_max) return false;
    *t_out = t;
    return true;
}

DEVFN int32_t d_trace_nearest(const DevScene& s, AV3 o, AV3 d,
                              float t_max, float* t_hit) {
    AV3 inv_d = av3(1.f / (d.x == 0.f ? 1e-30f : d.x),
                    1.f / (d.y == 0.f ? 1e-30f : d.y),
                    1.f / (d.z == 0.f ? 1e-30f : d.z));
    int32_t stack[AUR_STACK];
    int sp = 0; stack[sp++] = 0;
    int32_t best = -1; float tb = t_max;
    while (sp) {
        int32_t ni = stack[--sp];
        const ABvhNode n = s.nodes[ni];
        if (!d_hit_aabb(n, o, inv_d, tb)) continue;
        if (n.count) {
            for (int32_t i = n.left; i < n.left + n.count; ++i) {
                float t;
                if (d_hit_tri(s, i, o, d, tb, &t)) { tb = t; best = i; }
            }
        } else {
            stack[sp++] = s.right[ni];
            stack[sp++] = n.left;
        }
    }
    *t_hit = tb;
    return best;
}

DEVFN bool d_trace_any(const DevScene& s, AV3 o, AV3 d, float t_max) {
    AV3 inv_d = av3(1.f / (d.x == 0.f ? 1e-30f : d.x),
                    1.f / (d.y == 0.f ? 1e-30f : d.y),
                    1.f / (d.z == 0.f ? 1e-30f : d.z));
    int32_t stack[AUR_STACK];
    int sp = 0; stack[sp++] = 0;
    while (sp) {
        int32_t ni = stack[--sp];
        const ABvhNode n = s.nodes[ni];
        if (!d_hit_aabb(n, o, inv_d, t_max)) continue;
        if (n.count) {
            for (int32_t i = n.left; i < n.left + n.count; ++i) {
                float t;
                if (d_hit_tri(s, i, o, d, t_max, &t)) return true;
            }
        } else {
            stack[sp++] = s.right[ni];
            stack[sp++] = n.left;
        }
    }
    return false;
}

DEVFN AV3 d_sample_lambert(APcg32& rng, AV3 n) {
    float u1 = rng.f01(), u2 = rng.f01();
    float r = sqrtf(u1), phi = 6.28318530718f * u2;
    float x = r * cosf(phi), y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.f, 1.f - u1));
    AV3 a = fabsf(n.x) > 0.9f ? av3(0, 1, 0) : av3(1, 0, 0);
    AV3 t = vnorm(vcross(a, n));
    AV3 b = vcross(n, t);
    return vnorm(vadd(vadd(vscale(t, x), vscale(b, y)), vscale(n, z)));
}

DEVFN AV3 d_sample_sphere(APcg32& rng) {
    float z = 1.f - 2.f * rng.f01();
    float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    float phi = 6.28318530718f * rng.f01();
    return av3(r * cosf(phi), r * sinf(phi), z);
}

/* Gaussian pair (Box-Muller): tracer anti-crack re-aiming */
DEVFN void d_gauss2(APcg32& rng, float* g1, float* g2) {
    float u1 = rng.f01(); if (u1 < 1e-12f) u1 = 1e-12f;
    float u2 = rng.f01();
    float r = sqrtf(-2.f * logf(u1));
    *g1 = r * cosf(6.28318530718f * u2);
    *g2 = r * sinf(6.28318530718f * u2);
}

/* Kernel parameters (POD, passed by value) */
struct KParams {
    const float* absorption;      /* device (GPU) or host (EMU)         */
    const float* scattering;
    const float* transmission;
    const float* dir_table;       /* nullptr if omni                    */
    int32_t n_mats, n_bands, n_rays, max_bounces, n_bins;
    int32_t spec_exclude_order, n_dir_samples;
    float max_time, inv_bin, c, energy_cap;
    float rcv_r2;                 /* squared detection sphere radius    */
    float det_w;                  /* 4/r^2: expectation = 1/d^2 lattice */
    float kair[16];
    AV3 src, rcv, fwd;
    int32_t has_dir;
    uint64_t seed;
    double inv_n;
    /* reflection-based scattering (aur_trace2): per triangle in BVH order */
    const float* rbs_l;
    const float* rbs_w;
    float band_hz[16];
    int32_t has_rbs;
};

/* reflection-based scattering (Christensen & Rindel 2005, eqs. 1-5)  */
DEVFN float rbs_sd(float f, float l, float w, float cos_i, float astar, float c) {
    float ws = fmaxf(w * cos_i, 1e-3f), ls = fmaxf(l, 1e-3f);
    float a = fmaxf(astar, 1e-6f);
    float kw = fminf(1.f, f / (c * a / (2.f * ws * ws)));
    float kl = fminf(1.f, f / (c * a / (2.f * ls * ls)));
    return 1.f - kw * kl;
}
DEVFN float rbs_combine(float sd, float ss) { return 1.f - (1.f - sd) * (1.f - ss); }
DEVFN float rbs_astar(float di, float dr) { return di * dr / fmaxf(2.f * (di + dr), 1e-12f); }

/* One ray, the physics of aur_native.cpp. A PCG32 stream per ray (CPU: per thread) and float
 * atomics (CPU: double per thread): statistically equal to the CPU, not bit for bit. The GPU
 * atomic order varies from run to run; EMU is deterministic. */
KERNFN void trace_one_ray(int32_t ri, const DevScene s, const KParams kp,
                         float* hist, float* dirs, double* stats3) {
    APcg32 rng;
    rng.init(kp.seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(ri + 1),
             77 + (uint64_t)ri);

    const int NB = kp.n_bands;
    float E[16];
    AV3 o = kp.src;
    AV3 d = d_sample_sphere(rng);
    if (kp.has_dir) {
        float ct = vdot(d, kp.fwd);
        if (ct > 1.f) ct = 1.f; else if (ct < -1.f) ct = -1.f;
        float u = acosf(ct) * (float)(kp.n_dir_samples - 1)
                  / 3.14159265358979f;
        int i0 = (int)u;
        if (i0 > kp.n_dir_samples - 2) i0 = kp.n_dir_samples - 2;
        if (i0 < 0) i0 = 0;
        float fr = u - (float)i0;
        for (int b = 0; b < NB; ++b) {
            const float* row = kp.dir_table + (size_t)b * kp.n_dir_samples;
            E[b] = row[i0] * (1.f - fr) + row[i0 + 1] * fr;
        }
    } else {
        for (int b = 0; b < NB; ++b) E[b] = 1.f;
    }
    float time = 0.f;
    int order = 0;
    bool pure_spec = true;
    bool spec_seg = true;      /* segment from a specular branch         */
    int  retries = 0;          /* anti-crack re-aiming (launch included) */
    float dspec = 0.f;         /* path since the last diffuse event      */
    float wsp[16];             /* rbs: unbiased per-band weight of a
                                  specular segment, (1 - s_b) / (1 - s_mean) */
    for (int b = 0; b < NB; ++b) wsp[b] = 1.f;
    double bounces = 0.0, escaped = 0.0, deposits = 0.0;
    const int K = kp.spec_exclude_order;
    const float cap = kp.energy_cap;

    for (int bounce = 0; bounce < kp.max_bounces; ++bounce) {
        float t_hit;
        int32_t tri = d_trace_nearest(s, o, d, 1e30f, &t_hit);
        bool miss = (tri < 0);

        /* numerical cracks: re-aim before giving up */
        if (miss && retries < 2) {
            ++retries;
            float g1, g2, g3, g4;
            d_gauss2(rng, &g1, &g2);
            d_gauss2(rng, &g3, &g4);
            d = vnorm(av3(d.x + 0.008f * g1, d.y + 0.008f * g2,
                          d.z + 0.008f * g3));
            continue;          /* costs one bounce, as in python         */
        }

        /* sphere detection of the specular energy, final segments only; pure specular chains
         * of order <= K (order 0 = the direct sound) are left to the image sources */
        float segl = miss ? fmaxf(0.f, kp.c * (kp.max_time - time)) : t_hit;
        if (segl > 0.f && spec_seg && !(pure_spec && order <= K)) {
            AV3 wv = vsub(kp.rcv, o);
            float tc = vdot(wv, d);
            if (tc > 0.f && tc < segl) {
                float m2 = vdot(wv, wv) - tc * tc;
                if (m2 < kp.rcv_r2) {
                    int bin = (int)((time + tc / kp.c) * kp.inv_bin);
                    if (bin >= 0 && bin < kp.n_bins) {
                        float wsum = 0.f;
                        for (int b = 0; b < NB; ++b) {
                            float dep = (float)((double)E[b]
                                * expf(-kp.kair[b] * tc)
                                * (double)kp.det_w * kp.inv_n);
                            if (kp.has_rbs) dep *= wsp[b];
                            aurAtomicAddF(&hist[(size_t)b * kp.n_bins + bin],
                                          dep);
                            wsum += dep;
                        }
                        /* DOA: energy flies along d, arrives from -d  */
                        aurAtomicAddF(&dirs[(size_t)bin * 3 + 0], -wsum * d.x);
                        aurAtomicAddF(&dirs[(size_t)bin * 3 + 1], -wsum * d.y);
                        aurAtomicAddF(&dirs[(size_t)bin * 3 + 2], -wsum * d.z);
                        deposits += 1.0;
                    }
                }
            }
        }
        if (miss) { escaped += 1.0; break; }

        for (int b = 0; b < NB; ++b) E[b] *= expf(-kp.kair[b] * t_hit);
        time += t_hit / kp.c;
        if (time > kp.max_time) break;

        AV3 hp = vadd(o, vscale(d, t_hit));
        AV3 n = s.ng[tri];
        if (vdot(n, d) > 0.f) n = vscale(n, -1.f);

        int32_t m = s.mat[tri];
        if (m < 0) m = 0;
        if (m >= kp.n_mats) m = kp.n_mats - 1;
        const float* alpha = kp.absorption   + (size_t)m * NB;
        const float* scat  = kp.scattering   + (size_t)m * NB;
        const float* tau   = kp.transmission + (size_t)m * NB;

        float Eref[16], s_mean = 0.f, e_max = 0.f;
        for (int b = 0; b < NB; ++b) {
            float r = 1.f - alpha[b] - tau[b];
            if (r < 0.f) r = 0.f;
            Eref[b] = E[b] * r;
            s_mean += scat[b];
            if (Eref[b] > e_max) e_max = Eref[b];
        }
        s_mean /= NB;
        float d_in = 0.f, cos_i = 0.f, tl = 0.f, tw = 0.f, s_c[16];
        if (kp.has_rbs) {          /* the ray: reflected = incident        */
            d_in = dspec + t_hit;
            cos_i = fabsf(vdot(n, d));
            tl = kp.rbs_l[tri];
            tw = kp.rbs_w[tri];
            float sm = 0.f;
            for (int b = 0; b < NB; ++b) {
                s_c[b] = rbs_combine(rbs_sd(kp.band_hz[b], tl, tw, cos_i,
                                            0.25f * d_in, kp.c), scat[b]);
                sm += s_c[b];
            }
            s_mean = sm / NB;
        }

        /* diffuse rain towards the receiver: only the scattered part s*E radiates (Lambert);
         * the specular part is caught by the sphere detection */
        AV3 to_r = vsub(kp.rcv, hp);
        float dR = vlen(to_r);
        if (dR > 1e-4f) {
            AV3 ur = vscale(to_r, 1.f / dR);
            float costh = vdot(n, ur);
            if (costh > 0.f) {
                int bin = (int)((time + dR / kp.c) * kp.inv_bin);
                if (bin >= 0 && bin < kp.n_bins) {
                    AV3 off = vadd(hp, vscale(n, AUR_RAY_EPS * 4.f));
                    if (!d_trace_any(s, off, ur, dR - AUR_RAY_EPS * 8.f)) {
                        float wsum = 0.f;
                        float geo = 4.f * costh / (dR * dR);
                        for (int b = 0; b < NB; ++b) {
                            float sv = scat[b];
                            if (kp.has_rbs)   /* the last leg's true length */
                                sv = rbs_combine(rbs_sd(kp.band_hz[b], tl, tw, cos_i,
                                                        rbs_astar(d_in, dR), kp.c),
                                                 scat[b]);
                            float dep = (float)((double)Eref[b] * sv * geo
                                       * expf(-kp.kair[b] * dR) * kp.inv_n);
                            aurAtomicAddF(&hist[(size_t)b * kp.n_bins + bin],
                                          dep);
                            wsum += dep;
                        }
                        aurAtomicAddF(&dirs[(size_t)bin * 3 + 0],
                                      wsum * (hp.x - kp.rcv.x) / dR);
                        aurAtomicAddF(&dirs[(size_t)bin * 3 + 1],
                                      wsum * (hp.y - kp.rcv.y) / dR);
                        aurAtomicAddF(&dirs[(size_t)bin * 3 + 2],
                                      wsum * (hp.z - kp.rcv.z) / dR);
                        deposits += 1.0;
                    }
                }
            }
        }

        bounces += 1.0;
        order += 1;
        for (int b = 0; b < NB; ++b) E[b] = Eref[b];

        if (e_max < cap) {
            const float p_surv = 0.25f;
            if (rng.f01() > p_surv) break;
            for (int b = 0; b < NB; ++b) E[b] *= (1.f / p_surv);
            if (e_max < cap * 1e-4f) break;
        }

        bool go_diff = (rng.f01() < s_mean);
        if (go_diff) {
            d = d_sample_lambert(rng, n);
            pure_spec = false;
        } else {
            d = vsub(d, vscale(n, 2.f * vdot(d, n)));
        }
        spec_seg = !go_diff;
        if (kp.has_rbs) {
            dspec = go_diff ? 0.f : d_in;
            if (!go_diff) {
                float q = 1.f / fmaxf(1.f - s_mean, 1e-6f);
                for (int b = 0; b < NB; ++b) wsp[b] = (1.f - s_c[b]) * q;
            }
        }
        o = vadd(hp, vscale(n, AUR_RAY_EPS * 4.f));
    }
    aurAtomicAddD(&stats3[0], bounces);
    aurAtomicAddD(&stats3[1], escaped);
    aurAtomicAddD(&stats3[2], deposits);
}

#if AUR_GPU
/* AUR_HIST_REPL histogram replicas, block b writing into b % R, merged on the host: atomics on
 * one shared histogram serialize on the hot early bins (48x slower than the CPU). */
#define AUR_HIST_REPL 64
GLOBALFN void aur_trace_kernel(DevScene s, KParams kp,
                               float* hist, float* dirs, double* stats3) {
    int32_t ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= kp.n_rays) return;
    const size_t nh = (size_t)kp.n_bands * kp.n_bins;
    const size_t nd = (size_t)kp.n_bins * 3;
    const int rep = blockIdx.x % AUR_HIST_REPL;
    trace_one_ray(ri, s, kp, hist + (size_t)rep * nh,
                  dirs + (size_t)rep * nd, stats3 + (size_t)rep * 3);
}
#endif

/* Host side: BVH and API, the algorithms of aur_native.cpp (raycasts stay on the host) */
namespace {

struct AABB {
    AV3 lo{ 1e30f,  1e30f,  1e30f};
    AV3 hi{-1e30f, -1e30f, -1e30f};
    void grow(AV3 p) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y);
        lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y);
        hi.z = std::max(hi.z, p.z);
    }
    void grow(const AABB& b) { grow(b.lo); grow(b.hi); }
    float area() const {
        AV3 d = vsub(hi, lo);
        if (d.x < 0 || d.y < 0 || d.z < 0) return 0.f;
        return 2.f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

static inline float compv(AV3 a, int i) {
    return i == 0 ? a.x : (i == 1 ? a.y : a.z);
}

constexpr int SAH_BINS = 12;
constexpr int LEAF_MAX = 4;

struct BuildTri { AABB box; AV3 centroid; int32_t idx; };

struct HostScene {
    std::vector<AV3> v0, e1, e2, ng;
    std::vector<int32_t> mat, src_index, right;
    std::vector<ABvhNode> nodes;
    int32_t n_tris = 0;
};

static int32_t build2(HostScene& s, std::vector<BuildTri>& tris,
                      int32_t begin, int32_t end, int depth) {
    AABB box, cbox;
    for (int32_t i = begin; i < end; ++i) {
        box.grow(tris[i].box); cbox.grow(tris[i].centroid);
    }
    int32_t node_idx = (int32_t)s.nodes.size();
    s.nodes.push_back({box.lo, 0, box.hi, 0});
    s.right.push_back(-1);
    int32_t count = end - begin;
    if (count <= LEAF_MAX || depth > 60) {
        s.nodes[node_idx].left = begin;
        s.nodes[node_idx].count = count;
        return node_idx;
    }
    AV3 cext = vsub(cbox.hi, cbox.lo);
    int axis = 0;
    if (cext.y > compv(cext, axis)) axis = 1;
    if (cext.z > compv(cext, axis)) axis = 2;
    float cmin = compv(cbox.lo, axis), cext_a = compv(cext, axis);
    int32_t mid = -1;
    if (cext_a > 1e-9f) {
        AABB bin_box[SAH_BINS]; int bin_cnt[SAH_BINS] = {0};
        float k = SAH_BINS * (1.f - 1e-6f) / cext_a;
        for (int32_t i = begin; i < end; ++i) {
            int b = (int)((compv(tris[i].centroid, axis) - cmin) * k);
            b = std::min(std::max(b, 0), SAH_BINS - 1);
            bin_box[b].grow(tris[i].box); bin_cnt[b]++;
        }
        AABB right_acc[SAH_BINS]; AABB acc;
        for (int b = SAH_BINS - 1; b > 0; --b) {
            acc.grow(bin_box[b]); right_acc[b] = acc;
        }
        float best = 1e30f; int best_bin = -1; AABB lacc; int32_t running = 0;
        for (int b = 0; b < SAH_BINS - 1; ++b) {
            lacc.grow(bin_box[b]); running += bin_cnt[b];
            if (running == 0 || running == count) continue;
            float cost = lacc.area() * running
                       + right_acc[b + 1].area() * (count - running);
            if (cost < best) { best = cost; best_bin = b; }
        }
        if (best_bin >= 0) {
            float split = cmin + (best_bin + 1) * cext_a / SAH_BINS;
            auto it = std::partition(tris.begin() + begin, tris.begin() + end,
                [&](const BuildTri& t) {
                    return compv(t.centroid, axis) < split;
                });
            mid = (int32_t)(it - tris.begin());
            if (mid == begin || mid == end) mid = -1;
        }
    }
    if (mid < 0) {
        mid = begin + count / 2;
        std::nth_element(tris.begin() + begin, tris.begin() + mid,
                         tris.begin() + end,
            [&](const BuildTri& a, const BuildTri& b) {
                return compv(a.centroid, axis) < compv(b.centroid, axis);
            });
    }
    int32_t l = build2(s, tris, begin, mid, depth + 1);
    int32_t r = build2(s, tris, mid, end, depth + 1);
    s.nodes[node_idx].left = l;
    s.right[node_idx] = r;
    return node_idx;
}

} /* namespace */

#if AUR_GPU
static char g_gpu_err[256] = "";
static void gpu_note(const char* site, cudaError_t e) {
    std::snprintf(g_gpu_err, sizeof(g_gpu_err), "%s: %s",
                  site, cudaGetErrorString(e));
    std::fprintf(stderr, "[aur_cuda] %s\n", g_gpu_err);
}
#endif

struct AurCudaScene {
    HostScene h;
    DevScene host_view;              /* host pointers (EMU + raycast)     */
#if AUR_GPU
    DevScene dev_view;               /* device pointers                   */
    void* dbufs[7] = {nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, nullptr};
    bool dev_ok = false;
#endif
};

static DevScene make_view(const HostScene& h) {
    DevScene v;
    v.nodes = h.nodes.data(); v.right = h.right.data();
    v.v0 = h.v0.data(); v.e1 = h.e1.data(); v.e2 = h.e2.data();
    v.ng = h.ng.data(); v.mat = h.mat.data();
    v.n_tris = h.n_tris;
    return v;
}

#if AUR_GPU
template <typename T>
static bool dev_upload(const char* site, void** slot,
                       const std::vector<T>& src, const T** dst) {
    size_t bytes = src.size() * sizeof(T);
    cudaError_t e = cudaMalloc(slot, bytes);
    if (e != cudaSuccess) { gpu_note(site, e); return false; }
    e = cudaMemcpy(*slot, src.data(), bytes,
                   cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { gpu_note(site, e); return false; }
    *dst = (const T*)*slot;
    return true;
}
#endif

extern "C" {

AUR_API AurScene* aur_scene_create(const float* verts, int32_t n_tris,
                                   const int32_t* mat_idx) {
    if (!verts || n_tris <= 0) return nullptr;
    auto* w = new AurCudaScene();
    HostScene& s = w->h;
    s.n_tris = n_tris;

    std::vector<BuildTri> btris((size_t)n_tris);
    for (int32_t i = 0; i < n_tris; ++i) {
        const float* p = verts + 9 * (size_t)i;
        AV3 a = av3(p[0], p[1], p[2]);
        AV3 b = av3(p[3], p[4], p[5]);
        AV3 c = av3(p[6], p[7], p[8]);
        AABB box; box.grow(a); box.grow(b); box.grow(c);
        btris[i] = {box, vscale(vadd(vadd(a, b), c), 1.f / 3.f), i};
    }
    s.nodes.reserve((size_t)n_tris * 2);
    s.right.reserve((size_t)n_tris * 2);
    build2(s, btris, 0, n_tris, 0);

    s.v0.resize(n_tris); s.e1.resize(n_tris); s.e2.resize(n_tris);
    s.ng.resize(n_tris); s.mat.resize(n_tris); s.src_index.resize(n_tris);
    for (int32_t i = 0; i < n_tris; ++i) {
        int32_t oi = btris[i].idx;
        const float* p = verts + 9 * (size_t)oi;
        AV3 a = av3(p[0], p[1], p[2]);
        AV3 b = av3(p[3], p[4], p[5]);
        AV3 c = av3(p[6], p[7], p[8]);
        s.v0[i] = a; s.e1[i] = vsub(b, a); s.e2[i] = vsub(c, a);
        s.ng[i] = vnorm(vcross(s.e1[i], s.e2[i]));
        s.mat[i] = mat_idx ? mat_idx[oi] : 0;
        s.src_index[i] = oi;
    }
    w->host_view = make_view(s);

#if AUR_GPU
    w->dev_view = w->host_view;
    cudaError_t ectx = cudaFree(0);   /* force context init */
    if (ectx != cudaSuccess) gpu_note("context init (cudaFree 0)",
                                      ectx);
    bool ok = (ectx == cudaSuccess)
           && dev_upload("nodes", &w->dbufs[0], s.nodes,
                         &w->dev_view.nodes)
           && dev_upload("right", &w->dbufs[1], s.right,
                         &w->dev_view.right)
           && dev_upload("v0", &w->dbufs[2], s.v0, &w->dev_view.v0)
           && dev_upload("e1", &w->dbufs[3], s.e1, &w->dev_view.e1)
           && dev_upload("e2", &w->dbufs[4], s.e2, &w->dev_view.e2)
           && dev_upload("ng", &w->dbufs[5], s.ng, &w->dev_view.ng)
           && dev_upload("mat", &w->dbufs[6], s.mat,
                         &w->dev_view.mat);
    w->dev_ok = ok;
#endif
    return reinterpret_cast<AurScene*>(w);
}

AUR_API void aur_scene_destroy(AurScene* scene) {
    if (!scene) return;
    auto* w = reinterpret_cast<AurCudaScene*>(scene);
#if AUR_GPU
    for (void* b : w->dbufs) if (b) cudaFree(b);
#endif
    delete w;
}

AUR_API int32_t aur_scene_node_count(const AurScene* scene) {
    if (!scene) return 0;
    return (int32_t)reinterpret_cast<const AurCudaScene*>(scene)
        ->h.nodes.size();
}

AUR_API int32_t aur_raycast(const AurScene* scene, const float o[3],
                            const float d[3], float t_max,
                            float* t_out, float n_out[3]) {
    if (!scene) return -1;
    const auto* w = reinterpret_cast<const AurCudaScene*>(scene);
    float t;
    int32_t i = d_trace_nearest(w->host_view, av3(o[0], o[1], o[2]),
                                av3(d[0], d[1], d[2]), t_max, &t);
    if (i < 0) return -1;
    if (t_out) *t_out = t;
    if (n_out) {
        n_out[0] = w->h.ng[i].x; n_out[1] = w->h.ng[i].y;
        n_out[2] = w->h.ng[i].z;
    }
    return w->h.src_index[i];
}

AUR_API int32_t aur_segment_transmission(const AurScene* scene,
                                         const float a[3], const float b[3],
                                         const float* transmission,
                                         int32_t n_mats, int32_t n_bands,
                                         float* out_trans) {
    for (int32_t k = 0; k < n_bands; ++k) out_trans[k] = 1.f;
    if (!scene) return 0;
    const auto* w = reinterpret_cast<const AurCudaScene*>(scene);
    AV3 A = av3(a[0], a[1], a[2]), B = av3(b[0], b[1], b[2]);
    AV3 d = vsub(B, A);
    float dist = vlen(d);
    if (dist < 1e-6f) return 0;
    d = vscale(d, 1.f / dist);
    AV3 o = A;
    float remaining = dist - AUR_RAY_EPS;
    int32_t crossings = 0;
    for (int guard = 0; guard < 256 && remaining > AUR_RAY_EPS; ++guard) {
        float t;
        int32_t i = d_trace_nearest(w->host_view, o, d, remaining, &t);
        if (i < 0) break;
        int32_t m = std::min(std::max(w->h.mat[i], 0), n_mats - 1);
        const float* tr = transmission + (size_t)m * n_bands;
        for (int32_t k = 0; k < n_bands; ++k) out_trans[k] *= tr[k];
        crossings++;
        o = vadd(o, vscale(d, t + AUR_RAY_EPS));
        remaining -= (t + AUR_RAY_EPS);
        float mx = 0.f;
        for (int32_t k = 0; k < n_bands; ++k) mx = std::max(mx, out_trans[k]);
        if (mx < 1e-7f) break;
    }
    return crossings;
}

/* 1 if a GPU answers (always 1 in EMU). */
AUR_API int32_t aur_cuda_available(void) {
#if AUR_GPU
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0 ? 1 : 0;
#else
    return 1;
#endif
}

static int32_t trace_cuda(const AurScene* scene, const AurTraceParams* p,
                          const AurRbs* rbs,
                          float* out_hist, float* out_dir,
                          double* out_stats) {
    if (!scene || !p || !out_hist || !out_dir) return 1;
    if (p->n_bands <= 0 || p->n_bands > 16) return 2;
    if (rbs && (!rbs->tri_l || !rbs->tri_w || !rbs->band_hz)) return 3;
    const auto* w = reinterpret_cast<const AurCudaScene*>(scene);

    KParams kp;
    /* rbs inputs in BVH order: the kernel indexes triangles directly */
    std::vector<float> rl, rw;
    kp.has_rbs = rbs ? 1 : 0;
    kp.rbs_l = kp.rbs_w = nullptr;
    if (rbs) {
        const size_t nt = w->h.src_index.size();
        rl.resize(nt); rw.resize(nt);
        for (size_t i = 0; i < nt; ++i) {
            rl[i] = rbs->tri_l[w->h.src_index[i]];
            rw[i] = rbs->tri_w[w->h.src_index[i]];
        }
        for (int b = 0; b < p->n_bands; ++b) kp.band_hz[b] = rbs->band_hz[b];
    }
    kp.n_mats = p->n_mats; kp.n_bands = p->n_bands;
    kp.n_rays = p->n_rays; kp.max_bounces = p->max_bounces;
    kp.n_bins = p->n_bins;
    kp.spec_exclude_order = std::max(0, p->spec_exclude_order);
    kp.max_time = p->max_time;
    kp.inv_bin = 1.f / p->bin_dt;
    kp.c = p->speed_of_sound;
    kp.energy_cap = std::max(1e-9f, p->energy_cap);
    {
        float r_det = std::max(1e-3f, p->rcv_radius);
        kp.rcv_r2 = r_det * r_det;
        kp.det_w = 4.f / (r_det * r_det);
    }
    kp.seed = p->seed;
    kp.inv_n = 1.0 / std::max(1, p->n_rays);
    for (int b = 0; b < kp.n_bands; ++b)
        kp.kair[b] = 0.2302585093f
                   * (p->air_db_per_m ? p->air_db_per_m[b] : 0.f);
    kp.src = av3(p->src[0], p->src[1], p->src[2]);
    kp.rcv = av3(p->rcv[0], p->rcv[1], p->rcv[2]);
    AV3 fwd = av3(p->src_forward[0], p->src_forward[1], p->src_forward[2]);
    float fl = vlen(fwd);
    kp.has_dir = (p->n_dir_samples > 1) && (p->dir_table != nullptr)
                 && (fl > 1e-6f);
    kp.fwd = kp.has_dir ? vscale(fwd, 1.f / fl) : av3(0, 0, 1);
    kp.n_dir_samples = kp.has_dir ? p->n_dir_samples : 0;

    const size_t nh = (size_t)kp.n_bands * kp.n_bins;
    const size_t nd = (size_t)kp.n_bins * 3;

#if AUR_GPU
    if (!w->dev_ok) return 20;
    float *d_hist = nullptr, *d_dirs = nullptr;
    double* d_stats = nullptr;
    float *d_abs = nullptr, *d_sct = nullptr, *d_tau = nullptr,
          *d_dir = nullptr, *d_rl = nullptr, *d_rw = nullptr;
    cudaError_t e = cudaSuccess;
    #define CK(x) do { e = (x); if (e != cudaSuccess) { gpu_note(#x, e); goto fail; } } while (0)
    CK(cudaMalloc(&d_hist, AUR_HIST_REPL * nh * sizeof(float)));
    CK(cudaMalloc(&d_dirs, AUR_HIST_REPL * nd * sizeof(float)));
    CK(cudaMalloc(&d_stats, AUR_HIST_REPL * 3 * sizeof(double)));
    CK(cudaMemset(d_hist, 0, AUR_HIST_REPL * nh * sizeof(float)));
    CK(cudaMemset(d_dirs, 0, AUR_HIST_REPL * nd * sizeof(float)));
    CK(cudaMemset(d_stats, 0, AUR_HIST_REPL * 3 * sizeof(double)));
    {
        size_t mb = (size_t)kp.n_mats * kp.n_bands * sizeof(float);
        CK(cudaMalloc(&d_abs, mb));
        CK(cudaMalloc(&d_sct, mb));
        CK(cudaMalloc(&d_tau, mb));
        CK(cudaMemcpy(d_abs, p->absorption, mb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sct, p->scattering, mb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tau, p->transmission, mb, cudaMemcpyHostToDevice));
        kp.absorption = d_abs; kp.scattering = d_sct; kp.transmission = d_tau;
        if (kp.has_dir) {
            size_t db = (size_t)kp.n_bands * kp.n_dir_samples * sizeof(float);
            CK(cudaMalloc(&d_dir, db));
            CK(cudaMemcpy(d_dir, p->dir_table, db, cudaMemcpyHostToDevice));
            kp.dir_table = d_dir;
        } else {
            kp.dir_table = nullptr;
        }
        if (rbs) {
            size_t tb = rl.size() * sizeof(float);
            CK(cudaMalloc(&d_rl, tb));
            CK(cudaMalloc(&d_rw, tb));
            CK(cudaMemcpy(d_rl, rl.data(), tb, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_rw, rw.data(), tb, cudaMemcpyHostToDevice));
            kp.rbs_l = d_rl; kp.rbs_w = d_rw;
        }
    }
    {
        int block = 256;
        int grid = (kp.n_rays + block - 1) / block;
        const bool dbg = std::getenv("AUR_GPU_DEBUG") != nullptr;
        cudaEvent_t ev0, ev1;
        if (dbg) { cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                   cudaEventRecord(ev0); }
        aur_trace_kernel<<<grid, block>>>(w->dev_view, kp,
                                          d_hist, d_dirs, d_stats);
        CK(cudaGetLastError());
        if (dbg) cudaEventRecord(ev1);
        CK(cudaDeviceSynchronize());
        if (dbg) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr,
                         "[aur_cuda] kernel: %.1f ms  (%d blocs x %d, "
                         "%d repliques)\n", ms, grid, block,
                         AUR_HIST_REPL);
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        }
    }
    {
        std::vector<float> hh((size_t)AUR_HIST_REPL * nh),
                           dd((size_t)AUR_HIST_REPL * nd);
        std::vector<double> stv((size_t)AUR_HIST_REPL * 3);
        double st[3] = {0, 0, 0};
        CK(cudaMemcpy(hh.data(), d_hist,
                      hh.size() * sizeof(float),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dd.data(), d_dirs,
                      dd.size() * sizeof(float),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(stv.data(), d_stats,
                      stv.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        std::memset(out_hist, 0, nh * sizeof(float));
        std::memset(out_dir, 0, nd * sizeof(float));
        for (int r = 0; r < AUR_HIST_REPL; ++r) {
            const float* ph = hh.data() + (size_t)r * nh;
            const float* pd = dd.data() + (size_t)r * nd;
            for (size_t i = 0; i < nh; ++i) out_hist[i] += ph[i];
            for (size_t i = 0; i < nd; ++i) out_dir[i] += pd[i];
            st[0] += stv[(size_t)r * 3 + 0];
            st[1] += stv[(size_t)r * 3 + 1];
            st[2] += stv[(size_t)r * 3 + 2];
        }
        if (out_stats) {
            out_stats[0] = (double)kp.n_rays;
            out_stats[1] = st[0] / std::max(1, kp.n_rays);
            out_stats[2] = st[1];
            out_stats[3] = st[2];
        }
    }
    e = cudaSuccess;
fail:
    if (d_hist) cudaFree(d_hist);
    if (d_dirs) cudaFree(d_dirs);
    if (d_stats) cudaFree(d_stats);
    if (d_abs) cudaFree(d_abs);
    if (d_sct) cudaFree(d_sct);
    if (d_tau) cudaFree(d_tau);
    if (d_dir) cudaFree(d_dir);
    if (d_rl) cudaFree(d_rl);
    if (d_rw) cudaFree(d_rw);
    #undef CK
    return e == cudaSuccess ? 0 : (100 + (int)e);
#else
    /* EMU: the same kernel, one ray after another */
    kp.absorption = p->absorption;
    kp.scattering = p->scattering;
    kp.transmission = p->transmission;
    kp.dir_table = kp.has_dir ? p->dir_table : nullptr;
    if (rbs) { kp.rbs_l = rl.data(); kp.rbs_w = rw.data(); }
    std::memset(out_hist, 0, nh * sizeof(float));
    std::memset(out_dir, 0, nd * sizeof(float));
    double st[3] = {0, 0, 0};
    for (int32_t ri = 0; ri < kp.n_rays; ++ri)
        trace_one_ray(ri, w->host_view, kp, out_hist, out_dir, st);
    if (out_stats) {
        out_stats[0] = (double)kp.n_rays;
        out_stats[1] = st[0] / std::max(1, kp.n_rays);
        out_stats[2] = st[1];
        out_stats[3] = st[2];
    }
    return 0;
#endif
}


AUR_API int32_t aur_trace(const AurScene* scene, const AurTraceParams* p,
                          float* out_hist, float* out_dir,
                          double* out_stats) {
    return trace_cuda(scene, p, nullptr, out_hist, out_dir, out_stats);
}

AUR_API int32_t aur_trace2(const AurScene* scene, const AurTraceParams* p,
                           const AurRbs* rbs, float* out_hist, float* out_dir,
                           double* out_stats) {
    return trace_cuda(scene, p, rbs, out_hist, out_dir, out_stats);
}


/* Low-frequency FDTD: one thread per cell, one launch per step */

/* Wall faces grouped by cell: one thread per wall cell sums its faces in table order, regular
 * then resonant, as the CPU and EMU loops do. No atomics: the drive is bit-reproducible. */
struct DriveCsr {
    std::vector<int32_t> cell, fi, ri;
    std::vector<int64_t> fp, rp;
    int64_t n() const { return (int64_t)cell.size(); }
};

static void drive_csr(int64_t nf, const int32_t* fc, int64_t nr, const int32_t* rc,
                      DriveCsr& o) {
    std::vector<int32_t> f(nf > 0 ? nf : 0), r(nr > 0 ? nr : 0);
    for (int64_t k = 0; k < nf; ++k) f[k] = (int32_t)k;
    for (int64_t k = 0; k < nr; ++k) r[k] = (int32_t)k;
    std::stable_sort(f.begin(), f.end(), [fc](int32_t a, int32_t b) { return fc[a] < fc[b]; });
    std::stable_sort(r.begin(), r.end(), [rc](int32_t a, int32_t b) { return rc[a] < rc[b]; });
    o.cell.clear(); o.fi.clear(); o.ri.clear(); o.fp.assign(1, 0); o.rp.assign(1, 0);
    o.fi.reserve(f.size()); o.ri.reserve(r.size());
    size_t i = 0, j = 0;
    while (i < f.size() || j < r.size()) {
        int32_t c = INT32_MAX;
        if (i < f.size()) c = fc[f[i]];
        if (j < r.size() && rc[r[j]] < c) c = rc[r[j]];
        o.cell.push_back(c);
        while (i < f.size() && fc[f[i]] == c) o.fi.push_back(f[i++]);
        while (j < r.size() && rc[r[j]] == c) o.ri.push_back(r[j++]);
        o.fp.push_back((int64_t)o.fi.size());
        o.rp.push_back((int64_t)o.ri.size());
    }
}

#if AUR_GPU
GLOBALFN void aur_fdtd_drive_csr(int64_t n_wc, const int32_t* w_cell, const int64_t* w_fp,
                                 const int32_t* w_fi, const int64_t* w_rp, const int32_t* w_ri,
                                 const float* f_l, const float* f_Asum, const float* phi,
                                 const float* r_A, const float* r_c1, const float* r_c2,
                                 const float* psiR, const float* p1, float* D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_wc) return;
    const int64_t c = w_cell[i];
    float acc = 0.f;
    if (w_fi)
        for (int64_t k = w_fp[i]; k < w_fp[i + 1]; ++k) {
            const int64_t f = w_fi[k];
            acc += f_Asum[f] * p1[c] - (f_l[2 * f] * phi[2 * f] + f_l[2 * f + 1] * phi[2 * f + 1]);
        }
    if (w_ri)
        for (int64_t k = w_rp[i]; k < w_rp[i + 1]; ++k) {
            const int64_t f = w_ri[k];
            acc += r_A[f] * p1[c] - (r_c1[f] * psiR[2 * f] + r_c2[f] * psiR[2 * f + 1]);
        }
    D[c] = acc;
}
#endif

#if AUR_GPU
GLOBALFN void aur_fdtd_step(const int32_t* nb, const float* c_now,
                            const float* c_old, const float* c_nb,
                            int64_t n_air, const float* p0, const float* p1,
                            float* p2) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_air) return;
    const int32_t* q = nb + 6 * i;
    float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
            + p1[q[3]] + p1[q[4]] + p1[q[5]];
    p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i];
}

static inline int io_blocks(int64_t n) { return (int)((n + 255) / 256); }

/* one thread per tap / receiver: launch over io_blocks(n) blocks, never one fixed-size block
   (taps and receivers beyond it would be dropped silently) */
GLOBALFN void aur_fdtd_inject(float* p2, const int32_t* src_idx,
                              const float* src_w, int32_t n_src, float amp) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_src) atomicAdd(&p2[src_idx[k]], amp * src_w[k]);
}

GLOBALFN void aur_fdtd_gather(const float* p2, const int32_t* rec_idx,
                              const float* rec_w, int32_t n_rec,
                              float* out_step) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rec) return;
    float acc = 0.f;
    for (int k = 0; k < 8; ++k)
        acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
    out_step[r] = acc;
}
#endif


#if AUR_GPU
GLOBALFN void aur_fdtd_drive(int64_t n_faces, const int32_t* f_cell,
                             const float* f_l, const float* f_Asum,
                             const float* phi, const float* p1, float* D) {
    int64_t f = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;
    const int64_t c = f_cell[f];
    const float d = f_Asum[f] * p1[c]
        - (f_l[2 * f] * phi[2 * f] + f_l[2 * f + 1] * phi[2 * f + 1]);
    atomicAdd(&D[c], d);
}

GLOBALFN void aur_fdtd_step2(const int32_t* nb, const float* c_now,
                             const float* c_old, const float* c_nb,
                             const float* c_drv, const float* D,
                             int64_t n_air, const float* p0,
                             const float* p1, float* p2) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_air) return;
    const int32_t* q = nb + 6 * i;
    float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
            + p1[q[3]] + p1[q[4]] + p1[q[5]];
    p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i]
            - c_drv[i] * D[i];
}

GLOBALFN void aur_fdtd_phi(int64_t n_faces, const int32_t* f_cell,
                           const float* f_e, const float* f_g,
                           const float* p2, const float* p1,
                           float* phi) {
    int64_t f = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;
    const float pc = 0.5f * (p2[f_cell[f]] + p1[f_cell[f]]);
    phi[2 * f]     = f_e[2 * f]     * phi[2 * f]     + f_g[2 * f]     * pc;
    phi[2 * f + 1] = f_e[2 * f + 1] * phi[2 * f + 1] + f_g[2 * f + 1] * pc;
}

/* resonant sections (panels): drive and psi propagation, as the native have_r loops */
GLOBALFN void aur_fdtd_drive_res(int64_t n_rfaces, const int32_t* r_cell,
                                 const float* r_A, const float* r_c1,
                                 const float* r_c2, const float* psiR,
                                 const float* p1, float* D) {
    int64_t f = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_rfaces) return;
    const int64_t c = r_cell[f];
    const float d = r_A[f] * p1[c]
        - (r_c1[f] * psiR[2 * f] + r_c2[f] * psiR[2 * f + 1]);
    atomicAdd(&D[c], d);
}

GLOBALFN void aur_fdtd_psir(int64_t n_rfaces, const int32_t* r_cell,
                            const float* r_E, const float* r_G,
                            const float* p2, const float* p1,
                            float* psiR) {
    int64_t f = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_rfaces) return;
    const int64_t c = r_cell[f];
    const float pc = 0.5f * (p2[c] + p1[c]);
    const float s1 = psiR[2 * f], s2 = psiR[2 * f + 1];
    psiR[2 * f]     = r_E[4 * f]     * s1 + r_E[4 * f + 1] * s2
                    + r_G[2 * f]     * pc;
    psiR[2 * f + 1] = r_E[4 * f + 2] * s1 + r_E[4 * f + 3] * s2
                    + r_G[2 * f + 1] * pc;
}

/* run4 stencil, as the native cell_range: 4th order where mask4, else 2nd order - c_drv*D */
GLOBALFN void aur_fdtd_step4(const int32_t* nb, const int32_t* nb2,
                             const uint8_t* mask4,
                             const float* c_now, const float* c_old,
                             const float* c_nb, const float* c_drv,
                             float c4c, float c4nb, float c4nb2,
                             int64_t n_air, int32_t have_f,
                             const float* D,
                             const float* p0, const float* p1,
                             float* p2) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_air) return;
    const int32_t* q = nb + 6 * i;
    float sq = p1[q[0]] + p1[q[1]] + p1[q[2]]
             + p1[q[3]] + p1[q[4]] + p1[q[5]];
    if (mask4[i]) {
        const int32_t* q2 = nb2 + 6 * i;
        float s2 = p1[q2[0]] + p1[q2[1]] + p1[q2[2]]
                 + p1[q2[3]] + p1[q2[4]] + p1[q2[5]];
        p2[i] = c4c * p1[i] + c4nb * sq + c4nb2 * s2 - p0[i];
    } else {
        float v = c_now[i] * p1[i] + c_nb[i] * sq + c_old[i] * p0[i];
        if (have_f) v -= c_drv[i] * D[i];
        p2[i] = v;
    }
}

/* conformal tangential apertures, CSR by cell (no atomics):
   p2[i] += sum_k l_k (p1[nb_k] - p1[i]) */
GLOBALFN void aur_fdtd_links(int64_t n_lrows, const int32_t* l_rows,
                             const int64_t* l_ptr, const int32_t* l_nb,
                             const float* l_k, const float* p1, float* p2) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_lrows) return;
    const int64_t i = l_rows[r];
    float acc = 0.f;
    for (int64_t k = l_ptr[r]; k < l_ptr[r + 1]; ++k)
        acc += l_k[k] * (p1[l_nb[k]] - p1[i]);
    p2[i] += acc;
}

GLOBALFN void aur_fdtd_cpml_psi(int64_t n_pml,
                                const int32_t* pml_cells,
                                const int32_t* pml_nbp,
                                const float* pml_b,
                                const float* pml_apsi,
                                const float* p1, float* psi) {
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pml) return;
    const float dh = p1[pml_nbp[k]] - p1[pml_cells[k]];
    psi[k] = pml_b[k] * psi[k] + pml_apsi[k] * dh;
}

/* CPML of ONE axis, entries g = o + k: a cell appears at most once per axis and the three
 * axis launches serialize on the stream, so p2 needs no atomics. */
GLOBALFN void aur_fdtd_cpml_apply(int64_t o, int64_t count,
                                  const int32_t* pml_cells,
                                  const int32_t* pml_nbm,
                                  const int32_t* pml_nbp,
                                  const int32_t* pml_locm,
                                  const float* pml_b,
                                  const float* pml_a,
                                  const float* psi, float* phiP,
                                  float lam2, const float* p1,
                                  float* p2) {
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const int64_t g = o + k;
    const int64_t lm = pml_locm[g];
    const float pm = (lm >= 0) ? psi[o + lm] : 0.f;
    const float dpsi = psi[g] - pm;
    const int64_t cc = pml_cells[g];
    const float d2 = p1[pml_nbp[g]] - 2.f * p1[cc] + p1[pml_nbm[g]];
    const float ph = pml_b[g] * phiP[g] + pml_a[g] * (d2 + dpsi);
    phiP[g] = ph;
    p2[cc] += lam2 * (dpsi + ph);
}
#endif

/* FDTD progress: the step being computed, for a host poller */
static volatile int64_t* g_fdtd_progress = nullptr;
extern "C" AUR_API void aur_fdtd_progress(int64_t* counter) {
    g_fdtd_progress = counter;
}


extern "C" AUR_API int32_t aur_fdtd_run2(
        const int32_t* nb, const float* c_now, const float* c_old,
        const float* c_nb, const float* c_drv, int64_t n_air,
        int64_t n_faces, const int32_t* f_cell,
        const float* f_e, const float* f_g, const float* f_l,
        const float* f_Asum,
        int64_t n_rfaces, const int32_t* r_cell,
        const float* r_A, const float* r_c1, const float* r_c2,
        const float* r_E, const float* r_G,
        const float* pulse, int32_t n_pulse,
        const int32_t* src_idx, const float* src_w, int32_t n_src,
        const int32_t* rec_idx, const float* rec_w, int32_t n_rec,
        int32_t n_steps, int32_t n_threads, float* out) {
    (void)n_threads;
    const size_t na1 = (size_t)n_air + 1;
#if AUR_GPU
    cudaError_t e = cudaSuccess;
    int32_t *d_nb = 0, *d_si = 0, *d_ri = 0, *d_fc = 0;
    float *d_cn = 0, *d_co = 0, *d_cb = 0, *d_cd = 0, *d_p0 = 0, *d_p1 = 0,
          *d_p2 = 0, *d_sw = 0, *d_rw = 0, *d_out = 0,
          *d_fe = 0, *d_fg = 0, *d_fl = 0, *d_fA = 0, *d_phi = 0, *d_D = 0,
          *d_rA = 0, *d_rc1 = 0, *d_rc2 = 0, *d_rE = 0, *d_rG = 0,
          *d_psiR = 0;
    int32_t *d_rc = 0;
    const int64_t nr = (n_rfaces > 0 && r_cell) ? n_rfaces : 0;
    DriveCsr W;
    drive_csr(n_faces, f_cell, nr, r_cell, W);
    int32_t *d_wc = 0, *d_wfi = 0, *d_wri = 0;
    int64_t *d_wfp = 0, *d_wrp = 0;
    const int64_t n_wc = W.n();
    #define FK(x) do { e = (x); if (e != cudaSuccess) goto f2fail; } while (0)
    #define FKUP(dst, v, T) do { if (!(v).empty()) { \
        FK(cudaMalloc(&dst, (v).size() * sizeof(T))); \
        FK(cudaMemcpy(dst, (v).data(), (v).size() * sizeof(T), cudaMemcpyHostToDevice)); } } while (0)
    FKUP(d_wc, W.cell, int32_t);
    FKUP(d_wfp, W.fp, int64_t);
    FKUP(d_wfi, W.fi, int32_t);
    FKUP(d_wrp, W.rp, int64_t);
    FKUP(d_wri, W.ri, int32_t);
    #undef FKUP
    FK(cudaMalloc(&d_nb, 6 * (size_t)n_air * sizeof(int32_t)));
    FK(cudaMalloc(&d_cn, n_air * sizeof(float)));
    FK(cudaMalloc(&d_co, n_air * sizeof(float)));
    FK(cudaMalloc(&d_cb, n_air * sizeof(float)));
    FK(cudaMalloc(&d_cd, n_air * sizeof(float)));
    FK(cudaMalloc(&d_D,  n_air * sizeof(float)));
    FK(cudaMalloc(&d_p0, na1 * sizeof(float)));
    FK(cudaMalloc(&d_p1, na1 * sizeof(float)));
    FK(cudaMalloc(&d_p2, na1 * sizeof(float)));
    FK(cudaMalloc(&d_si, (size_t)n_src * sizeof(int32_t)));
    FK(cudaMalloc(&d_sw, (size_t)n_src * sizeof(float)));
    FK(cudaMalloc(&d_ri, 8 * (size_t)n_rec * sizeof(int32_t)));
    FK(cudaMalloc(&d_rw, 8 * (size_t)n_rec * sizeof(float)));
    FK(cudaMalloc(&d_out, (size_t)n_steps * n_rec * sizeof(float)));
    FK(cudaMalloc(&d_fc, (size_t)n_faces * sizeof(int32_t)));
    FK(cudaMalloc(&d_fe, 2 * (size_t)n_faces * sizeof(float)));
    FK(cudaMalloc(&d_fg, 2 * (size_t)n_faces * sizeof(float)));
    FK(cudaMalloc(&d_fl, 2 * (size_t)n_faces * sizeof(float)));
    FK(cudaMalloc(&d_fA, (size_t)n_faces * sizeof(float)));
    FK(cudaMalloc(&d_phi, 2 * (size_t)n_faces * sizeof(float)));
    if (nr) {
        FK(cudaMalloc(&d_rc, (size_t)nr * sizeof(int32_t)));
        FK(cudaMalloc(&d_rA, (size_t)nr * sizeof(float)));
        FK(cudaMalloc(&d_rc1, (size_t)nr * sizeof(float)));
        FK(cudaMalloc(&d_rc2, (size_t)nr * sizeof(float)));
        FK(cudaMalloc(&d_rE, 4 * (size_t)nr * sizeof(float)));
        FK(cudaMalloc(&d_rG, 2 * (size_t)nr * sizeof(float)));
        FK(cudaMalloc(&d_psiR, 2 * (size_t)nr * sizeof(float)));
        FK(cudaMemcpy(d_rc, r_cell, (size_t)nr * sizeof(int32_t),
                      cudaMemcpyHostToDevice));
        FK(cudaMemcpy(d_rA, r_A, (size_t)nr * sizeof(float),
                      cudaMemcpyHostToDevice));
        FK(cudaMemcpy(d_rc1, r_c1, (size_t)nr * sizeof(float),
                      cudaMemcpyHostToDevice));
        FK(cudaMemcpy(d_rc2, r_c2, (size_t)nr * sizeof(float),
                      cudaMemcpyHostToDevice));
        FK(cudaMemcpy(d_rE, r_E, 4 * (size_t)nr * sizeof(float),
                      cudaMemcpyHostToDevice));
        FK(cudaMemcpy(d_rG, r_G, 2 * (size_t)nr * sizeof(float),
                      cudaMemcpyHostToDevice));
        FK(cudaMemset(d_psiR, 0, 2 * (size_t)nr * sizeof(float)));
    }
    FK(cudaMemcpy(d_nb, nb, 6 * (size_t)n_air * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_cn, c_now, n_air * sizeof(float), cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_co, c_old, n_air * sizeof(float), cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_cb, c_nb, n_air * sizeof(float), cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_cd, c_drv, n_air * sizeof(float), cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_si, src_idx, n_src * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_sw, src_w, n_src * sizeof(float), cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_ri, rec_idx, 8 * (size_t)n_rec * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_rw, rec_w, 8 * (size_t)n_rec * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_fc, f_cell, (size_t)n_faces * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_fe, f_e, 2 * (size_t)n_faces * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_fg, f_g, 2 * (size_t)n_faces * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_fl, f_l, 2 * (size_t)n_faces * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_fA, f_Asum, (size_t)n_faces * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemset(d_p0, 0, na1 * sizeof(float)));
    FK(cudaMemset(d_p1, 0, na1 * sizeof(float)));
    FK(cudaMemset(d_p2, 0, na1 * sizeof(float)));
    FK(cudaMemset(d_phi, 0, 2 * (size_t)n_faces * sizeof(float)));
    FK(cudaMemset(d_D, 0, n_air * sizeof(float)));   /* only wall cells are written */
    {
        int block = 256;
        int gridc = (int)((n_air + block - 1) / block);
        int gridf = (int)((n_faces + block - 1) / block);
        for (int32_t it = 0; it < n_steps; ++it) {
            if (g_fdtd_progress) *g_fdtd_progress = it;
            if (n_wc > 0)
                aur_fdtd_drive_csr<<<(int)((n_wc + block - 1) / block), block>>>(
                    n_wc, d_wc, d_wfp, d_wfi, d_wrp, d_wri, d_fl, d_fA, d_phi, d_rA, d_rc1,
                    d_rc2, d_psiR, d_p1, d_D);
            aur_fdtd_step2<<<gridc, block>>>(d_nb, d_cn, d_co, d_cb, d_cd,
                                             d_D, n_air, d_p0, d_p1, d_p2);
            if (it < n_pulse && n_src > 0)
                aur_fdtd_inject<<<io_blocks(n_src), 256>>>(d_p2, d_si, d_sw, n_src,
                                                           pulse[it]);
            if (n_rec > 0)
                aur_fdtd_gather<<<io_blocks(n_rec), 256>>>(d_p2, d_ri, d_rw, n_rec,
                                                           d_out + (size_t)it * n_rec);
            if (n_faces)
                aur_fdtd_phi<<<gridf, block>>>(n_faces, d_fc, d_fe, d_fg,
                                               d_p2, d_p1, d_phi);
            if (nr)
                aur_fdtd_psir<<<(int)((nr + block - 1) / block),
                                block>>>(nr, d_rc, d_rE, d_rG,
                                         d_p2, d_p1, d_psiR);
            float* t0 = d_p0; d_p0 = d_p1; d_p1 = d_p2; d_p2 = t0;
        }
        FK(cudaGetLastError());
        FK(cudaDeviceSynchronize());
        FK(cudaMemcpy(out, d_out, (size_t)n_steps * n_rec * sizeof(float),
                      cudaMemcpyDeviceToHost));
    }
    e = cudaSuccess;
f2fail:
    if (d_nb) cudaFree(d_nb);
    if (d_cn) cudaFree(d_cn);
    if (d_co) cudaFree(d_co);
    if (d_cb) cudaFree(d_cb);
    if (d_cd) cudaFree(d_cd);
    if (d_wc) cudaFree(d_wc);
    if (d_wfp) cudaFree(d_wfp);
    if (d_wfi) cudaFree(d_wfi);
    if (d_wrp) cudaFree(d_wrp);
    if (d_wri) cudaFree(d_wri);
    if (d_D) cudaFree(d_D);
    if (d_p0) cudaFree(d_p0);
    if (d_p1) cudaFree(d_p1);
    if (d_p2) cudaFree(d_p2);
    if (d_si) cudaFree(d_si);
    if (d_sw) cudaFree(d_sw);
    if (d_ri) cudaFree(d_ri);
    if (d_rw) cudaFree(d_rw);
    if (d_out) cudaFree(d_out);
    if (d_rc) cudaFree(d_rc);
    if (d_rA) cudaFree(d_rA);
    if (d_rc1) cudaFree(d_rc1);
    if (d_rc2) cudaFree(d_rc2);
    if (d_rE) cudaFree(d_rE);
    if (d_rG) cudaFree(d_rG);
    if (d_psiR) cudaFree(d_psiR);
    if (d_fc) cudaFree(d_fc);
    if (d_fe) cudaFree(d_fe);
    if (d_fg) cudaFree(d_fg);
    if (d_fl) cudaFree(d_fl);
    if (d_fA) cudaFree(d_fA);
    if (d_phi) cudaFree(d_phi);
    #undef FK
    return e == cudaSuccess ? 0 : (200 + (int)e);
#else
    /* EMU: sequential, same order as the native CPU path */
    std::vector<float> P0(na1, 0.f), P1(na1, 0.f), P2(na1, 0.f);
    std::vector<float> D((size_t)n_air, 0.f);
    std::vector<float> phi(2 * (size_t)n_faces, 0.f);
    const bool have_re = (n_rfaces > 0 && r_cell);
    std::vector<float> psiR(have_re ? 2 * (size_t)n_rfaces : 0, 0.f);
    float* p0 = P0.data(); float* p1 = P1.data(); float* p2 = P2.data();
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        std::fill(D.begin(), D.end(), 0.f);
        for (int64_t f = 0; f < n_faces; ++f) {
            const int64_t c = f_cell[f];
            D[c] += f_Asum[f] * p1[c]
                  - (f_l[2 * f] * phi[2 * f] + f_l[2 * f + 1] * phi[2 * f + 1]);
        }
        if (have_re)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                D[c] += r_A[f] * p1[c]
                      - (r_c1[f] * psiR[2 * f]
                         + r_c2[f] * psiR[2 * f + 1]);
            }
        for (int64_t i = 0; i < n_air; ++i) {
            const int32_t* q = nb + 6 * i;
            float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                    + p1[q[3]] + p1[q[4]] + p1[q[5]];
            p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i]
                    - c_drv[i] * D[i];
        }
        p2[n_air] = 0.f;
        if (it < n_pulse)
            for (int32_t k = 0; k < n_src; ++k)
                p2[src_idx[k]] += pulse[it] * src_w[k];
        for (int32_t r = 0; r < n_rec; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 8; ++k)
                acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
            out[(size_t)it * n_rec + r] = acc;
        }
        for (int64_t f = 0; f < n_faces; ++f) {
            const float pc = 0.5f * (p2[f_cell[f]] + p1[f_cell[f]]);
            phi[2 * f]     = f_e[2 * f]     * phi[2 * f]     + f_g[2 * f] * pc;
            phi[2 * f + 1] = f_e[2 * f + 1] * phi[2 * f + 1]
                           + f_g[2 * f + 1] * pc;
        }
        if (have_re)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                const float pc = 0.5f * (p2[c] + p1[c]);
                const float s1 = psiR[2 * f], s2 = psiR[2 * f + 1];
                psiR[2 * f]     = r_E[4 * f]     * s1 + r_E[4 * f + 1] * s2
                                + r_G[2 * f]     * pc;
                psiR[2 * f + 1] = r_E[4 * f + 2] * s1 + r_E[4 * f + 3] * s2
                                + r_G[2 * f + 1] * pc;
            }
        float* tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
    }
    return 0;
#endif
}

extern "C" AUR_API int32_t aur_fdtd_run(
        const int32_t* nb, const float* c_now, const float* c_old,
        const float* c_nb, int64_t n_air,
        const float* pulse, int32_t n_pulse,
        const int32_t* src_idx, const float* src_w, int32_t n_src,
        const int32_t* rec_idx, const float* rec_w, int32_t n_rec,
        int32_t n_steps, int32_t n_threads, float* out) {
    (void)n_threads;
    if (!nb || !c_now || n_air <= 0 || n_steps <= 0 || !out) return 1;
#if AUR_GPU
    int32_t *d_nb = 0, *d_si = 0, *d_ri = 0;
    float *d_cn = 0, *d_co = 0, *d_cb = 0, *d_sw = 0, *d_rw = 0;
    float *d_p0 = 0, *d_p1 = 0, *d_p2 = 0, *d_out = 0;
    cudaError_t e = cudaSuccess;
    size_t na1 = (size_t)n_air + 1;
    #define FK(x) do { e = (x); if (e != cudaSuccess) goto ffail; } while (0)
    FK(cudaMalloc(&d_nb, 6 * (size_t)n_air * sizeof(int32_t)));
    FK(cudaMalloc(&d_cn, n_air * sizeof(float)));
    FK(cudaMalloc(&d_co, n_air * sizeof(float)));
    FK(cudaMalloc(&d_cb, n_air * sizeof(float)));
    FK(cudaMalloc(&d_p0, na1 * sizeof(float)));
    FK(cudaMalloc(&d_p1, na1 * sizeof(float)));
    FK(cudaMalloc(&d_p2, na1 * sizeof(float)));
    FK(cudaMalloc(&d_si, (size_t)n_src * sizeof(int32_t)));
    FK(cudaMalloc(&d_sw, (size_t)n_src * sizeof(float)));
    FK(cudaMalloc(&d_ri, 8 * (size_t)n_rec * sizeof(int32_t)));
    FK(cudaMalloc(&d_rw, 8 * (size_t)n_rec * sizeof(float)));
    FK(cudaMalloc(&d_out, (size_t)n_steps * n_rec * sizeof(float)));
    FK(cudaMemcpy(d_nb, nb, 6 * (size_t)n_air * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_cn, c_now, n_air * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_co, c_old, n_air * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_cb, c_nb, n_air * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_si, src_idx, n_src * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_sw, src_w, n_src * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_ri, rec_idx, 8 * (size_t)n_rec * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
    FK(cudaMemcpy(d_rw, rec_w, 8 * (size_t)n_rec * sizeof(float),
                  cudaMemcpyHostToDevice));
    FK(cudaMemset(d_p0, 0, na1 * sizeof(float)));
    FK(cudaMemset(d_p1, 0, na1 * sizeof(float)));
    FK(cudaMemset(d_p2, 0, na1 * sizeof(float)));
    {
        int block = 256;
        int grid = (int)((n_air + block - 1) / block);
        for (int32_t it = 0; it < n_steps; ++it) {
            if (g_fdtd_progress) *g_fdtd_progress = it;
            aur_fdtd_step<<<grid, block>>>(d_nb, d_cn, d_co, d_cb, n_air,
                                           d_p0, d_p1, d_p2);
            if (it < n_pulse && n_src > 0)
                aur_fdtd_inject<<<io_blocks(n_src), 256>>>(d_p2, d_si, d_sw, n_src,
                                                           pulse[it]);
            if (n_rec > 0)
                aur_fdtd_gather<<<io_blocks(n_rec), 256>>>(d_p2, d_ri, d_rw, n_rec,
                                                           d_out + (size_t)it * n_rec);
            float* t0 = d_p0; d_p0 = d_p1; d_p1 = d_p2; d_p2 = t0;
        }
        FK(cudaGetLastError());
        FK(cudaDeviceSynchronize());
        FK(cudaMemcpy(out, d_out, (size_t)n_steps * n_rec * sizeof(float),
                      cudaMemcpyDeviceToHost));
    }
    e = cudaSuccess;
ffail:
    if (d_nb) cudaFree(d_nb);
    if (d_cn) cudaFree(d_cn);
    if (d_co) cudaFree(d_co);
    if (d_cb) cudaFree(d_cb);
    if (d_p0) cudaFree(d_p0);
    if (d_p1) cudaFree(d_p1);
    if (d_p2) cudaFree(d_p2);
    if (d_si) cudaFree(d_si);
    if (d_sw) cudaFree(d_sw);
    if (d_ri) cudaFree(d_ri);
    if (d_rw) cudaFree(d_rw);
    if (d_out) cudaFree(d_out);
    #undef FK
    return e == cudaSuccess ? 0 : (200 + (int)e);
#else
    std::vector<float> P0((size_t)n_air + 1, 0.f),
                       P1((size_t)n_air + 1, 0.f),
                       P2((size_t)n_air + 1, 0.f);
    float* p0 = P0.data(); float* p1 = P1.data(); float* p2 = P2.data();
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        for (int64_t i = 0; i < n_air; ++i) {
            const int32_t* q = nb + 6 * i;
            float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                    + p1[q[3]] + p1[q[4]] + p1[q[5]];
            p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i];
        }
        p2[n_air] = 0.f;
        if (it < n_pulse)
            for (int32_t k = 0; k < n_src; ++k)
                p2[src_idx[k]] += pulse[it] * src_w[k];
        for (int32_t r = 0; r < n_rec; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 8; ++k)
                acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
            out[(size_t)it * n_rec + r] = acc;
        }
        float* t = p0; p0 = p1; p1 = p2; p2 = t;
    }
    return 0;
#endif
}

AUR_API const char* aur_last_gpu_error(void) {
#if AUR_GPU
    return g_gpu_err;
#else
    return "";
#endif
}

AUR_API int32_t aur_dev_ok(void* handle) {
#if AUR_GPU
    AurCudaScene* w = (AurCudaScene*)handle;
    return (handle && w->dev_ok) ? 1 : 0;
#else
    return handle ? 1 : 0;
#endif
}

/* run4 on one GPU, same signature as the native one: arrays uploaded once, per-step launches
 * without host sync, probes copied back at the end. 1 bad arguments, 30 no CUDA context,
 * 31 CUDA error, 40 debug divergence, 41 instability. */
static int32_t fdtd_run4_impl(const int32_t* nb, const int32_t* nb2,
                              const uint8_t* mask4,
                              const float* c_now, const float* c_old,
                              const float* c_nb, const float* c_drv,
                              float c4c, float c4nb, float c4nb2,
                              float lam2, int64_t n_air,
                              int64_t n_faces, const int32_t* f_cell,
                              const float* f_e, const float* f_g,
                              const float* f_l, const float* f_Asum,
                              int64_t n_rfaces, const int32_t* r_cell,
                              const float* r_A, const float* r_c1,
                              const float* r_c2, const float* r_E,
                              const float* r_G,
                              const int64_t* pml_counts,
                              const int32_t* pml_cells,
                              const float* pml_b, const float* pml_a,
                              const float* pml_apsi,
                              const int32_t* pml_nbm,
                              const int32_t* pml_nbp,
                              const int32_t* pml_locm,
                              int64_t n_lrows, const int32_t* l_rows,
                              const int64_t* l_ptr, const int32_t* l_nb,
                              const float* l_k,
                              const float* pulse, int32_t n_pulse,
                              const int32_t* src_idx, const float* src_w,
                              int32_t n_src,
                              const int32_t* rec_idx, const float* rec_w,
                              int32_t n_rec,
                              int32_t n_steps, int32_t n_threads,
                              float* out) {
    (void)n_threads;
    if (!nb || !nb2 || !mask4 || !c_now || n_air <= 0 || n_steps <= 0
        || !out) return 1;
#if AUR_GPU
    cudaError_t e = cudaFree(0);
    if (e != cudaSuccess) { gpu_note("fdtd context init", e); return 30; }

    int64_t off[4] = {0, 0, 0, 0};
    int64_t n_pml = 0;
    if (pml_counts) {
        for (int a = 0; a < 3; ++a) { off[a] = n_pml;
                                      n_pml += pml_counts[a]; }
        off[3] = n_pml;
    }
    const bool have_f = (n_faces > 0 && f_cell && c_drv);
    const bool have_r = (n_rfaces > 0 && r_cell && c_drv);
    const bool have_d = (have_f || have_r);

    std::vector<void*> owned;
    #define UP(dst, src, n, T) do { \
        void* _p = nullptr; size_t _b = (size_t)(n) * sizeof(T); \
        if (_b) { \
            e = cudaMalloc(&_p, _b); \
            if (e != cudaSuccess) { gpu_note("fdtd cudaMalloc " #dst, e); goto fail; } \
            owned.push_back(_p); \
            e = cudaMemcpy(_p, (src), _b, cudaMemcpyHostToDevice); \
            if (e != cudaSuccess) { gpu_note("fdtd upload " #dst, e); goto fail; } \
        } \
        dst = (T*)_p; } while (0)
    #define AZ(dst, n, T) do { \
        void* _p = nullptr; size_t _b = (size_t)(n) * sizeof(T); \
        if (_b) { \
            e = cudaMalloc(&_p, _b); \
            if (e != cudaSuccess) { gpu_note("fdtd cudaMalloc " #dst, e); goto fail; } \
            owned.push_back(_p); \
            e = cudaMemset(_p, 0, _b); \
            if (e != cudaSuccess) { gpu_note("fdtd memset " #dst, e); goto fail; } \
        } \
        dst = (T*)_p; } while (0)

    {
    const int32_t* d_nb = nullptr;  const int32_t* d_nb2 = nullptr;
    const uint8_t* d_m4 = nullptr;
    const float* d_cn = nullptr; const float* d_co = nullptr;
    const float* d_cb = nullptr; const float* d_cd = nullptr;
    const int32_t* d_fc = nullptr;
    const float* d_fe = nullptr; const float* d_fg = nullptr;
    const float* d_fl = nullptr; const float* d_fA = nullptr;
    const int32_t* d_pc = nullptr; const int32_t* d_pm = nullptr;
    const int32_t* d_pp = nullptr; const int32_t* d_pl = nullptr;
    const float* d_pb = nullptr; const float* d_pa = nullptr;
    const float* d_ps = nullptr;
    const int32_t* d_si = nullptr; const float* d_sw = nullptr;
    const int32_t* d_ri = nullptr; const float* d_rw = nullptr;
    float* d_p0 = nullptr; float* d_p1 = nullptr; float* d_p2 = nullptr;
    float* d_D = nullptr; float* d_phiF = nullptr;
    float* d_psi = nullptr; float* d_phiP = nullptr;
    float* d_out = nullptr;
    const int32_t* d_rc = nullptr;
    const float* d_rA = nullptr; const float* d_rc1 = nullptr;
    const float* d_rc2 = nullptr; const float* d_rE = nullptr;
    const float* d_rG = nullptr; float* d_psiR = nullptr;
    const int32_t* d_lr = nullptr; const int64_t* d_lp = nullptr;
    const int32_t* d_ln = nullptr; const float* d_lk = nullptr;
    const int32_t* d_wc = nullptr; const int64_t* d_wfp = nullptr;
    const int32_t* d_wfi = nullptr; const int64_t* d_wrp = nullptr;
    const int32_t* d_wri = nullptr;
    int64_t n_wc = 0;

    UP(d_nb, nb, 6 * n_air, int32_t);
    UP(d_nb2, nb2, 6 * n_air, int32_t);
    UP(d_m4, mask4, n_air, uint8_t);
    UP(d_cn, c_now, n_air, float);
    UP(d_co, c_old, n_air, float);
    UP(d_cb, c_nb, n_air, float);
    if (have_d) {
        UP(d_cd, c_drv, n_air, float);
        AZ(d_D, n_air, float);
    }
    if (have_f) {
        UP(d_fc, f_cell, n_faces, int32_t);
        UP(d_fe, f_e, 2 * n_faces, float);
        UP(d_fg, f_g, 2 * n_faces, float);
        UP(d_fl, f_l, 2 * n_faces, float);
        UP(d_fA, f_Asum, n_faces, float);
        AZ(d_phiF, 2 * n_faces, float);
    }
    if (have_r) {
        UP(d_rc, r_cell, n_rfaces, int32_t);
        UP(d_rA, r_A, n_rfaces, float);
        UP(d_rc1, r_c1, n_rfaces, float);
        UP(d_rc2, r_c2, n_rfaces, float);
        UP(d_rE, r_E, 4 * n_rfaces, float);
        UP(d_rG, r_G, 2 * n_rfaces, float);
        AZ(d_psiR, 2 * n_rfaces, float);
    }
    if (have_d) {
        DriveCsr W;
        drive_csr(have_f ? n_faces : 0, f_cell, have_r ? n_rfaces : 0, r_cell, W);
        n_wc = W.n();
        UP(d_wc, W.cell.data(), W.cell.size(), int32_t);
        UP(d_wfp, W.fp.data(), W.fp.size(), int64_t);
        UP(d_wfi, W.fi.data(), W.fi.size(), int32_t);
        UP(d_wrp, W.rp.data(), W.rp.size(), int64_t);
        UP(d_wri, W.ri.data(), W.ri.size(), int32_t);
    }
    if (n_pml > 0) {
        UP(d_pc, pml_cells, n_pml, int32_t);
        UP(d_pm, pml_nbm, n_pml, int32_t);
        UP(d_pp, pml_nbp, n_pml, int32_t);
        UP(d_pl, pml_locm, n_pml, int32_t);
        UP(d_pb, pml_b, n_pml, float);
        UP(d_pa, pml_a, n_pml, float);
        UP(d_ps, pml_apsi, n_pml, float);
        AZ(d_psi, n_pml, float);
        AZ(d_phiP, n_pml, float);
    }
    if (n_lrows > 0) {
        UP(d_lr, l_rows, n_lrows, int32_t);
        UP(d_lp, l_ptr, n_lrows + 1, int64_t);
        UP(d_ln, l_nb, l_ptr[n_lrows], int32_t);
        UP(d_lk, l_k, l_ptr[n_lrows], float);
    }
    if (n_src > 0) { UP(d_si, src_idx, n_src, int32_t);
                     UP(d_sw, src_w, n_src, float); }
    if (n_rec > 0) { UP(d_ri, rec_idx, 8 * n_rec, int32_t);
                     UP(d_rw, rec_w, 8 * n_rec, float); }
    AZ(d_p0, n_air + 1, float);
    AZ(d_p1, n_air + 1, float);
    AZ(d_p2, n_air + 1, float);
    AZ(d_out, (size_t)n_steps * (n_rec > 0 ? n_rec : 1), float);

    const bool timing = std::getenv("AUR_FDTD_TIMING") != nullptr;
    auto t_b = std::chrono::steady_clock::now();
    const int B = 256;
    const int gc = (int)((n_air + B - 1) / B);
    const int gf = have_f ? (int)((n_faces + B - 1) / B) : 0;
    const int gp = n_pml > 0 ? (int)((n_pml + B - 1) / B) : 0;

    /* AUR_FDTD_DEBUG=N: every N steps, scan the live arrays and stop at the first non-finite */
    int dbg_every = 0;
    { const char* dv = std::getenv("AUR_FDTD_DEBUG");
      if (dv) dbg_every = std::atoi(dv); }
    /* growth sentinel: a strided sample of |p| every 512 steps catches exponential
       instabilities (GKS edge modes of open domains) at ~1e4x, before inf */
    const int sent_every = 512;
    float sent_ref = 0.f;
    std::vector<float> sent_buf(25000);
    std::vector<float> dbg_buf;
    if (dbg_every > 0)
        dbg_buf.resize((size_t)n_air + 1);

    for (int32_t it = 0; it < n_steps; ++it) {

        if (g_fdtd_progress) *g_fdtd_progress = it;
        if (have_d && n_wc > 0)        /* D of the other cells stays 0 (zeroed once) */
            aur_fdtd_drive_csr<<<(int)((n_wc + B - 1) / B), B>>>(
                n_wc, d_wc, d_wfp, d_wfi, d_wrp, d_wri, d_fl, d_fA, d_phiF, d_rA, d_rc1, d_rc2,
                d_psiR, d_p1, d_D);
        aur_fdtd_step4<<<gc, B>>>(d_nb, d_nb2, d_m4, d_cn, d_co, d_cb,
                                  d_cd, c4c, c4nb, c4nb2, n_air,
                                  have_d ? 1 : 0, d_D, d_p0, d_p1,
                                  d_p2);
        if (n_lrows > 0)
            aur_fdtd_links<<<(int)((n_lrows + B - 1) / B), B>>>(
                n_lrows, d_lr, d_lp, d_ln, d_lk, d_p1, d_p2);
        if (n_pml > 0) {
            aur_fdtd_cpml_psi<<<gp, B>>>(n_pml, d_pc, d_pp, d_pb,
                                         d_ps, d_p1, d_psi);
            for (int a = 0; a < 3; ++a) {
                const int64_t cnt = off[a + 1] - off[a];
                if (cnt <= 0) continue;
                const int ga = (int)((cnt + B - 1) / B);
                aur_fdtd_cpml_apply<<<ga, B>>>(off[a], cnt, d_pc, d_pm,
                                               d_pp, d_pl, d_pb, d_pa,
                                               d_psi, d_phiP, lam2,
                                               d_p1, d_p2);
            }
        }
        cudaMemsetAsync(d_p2 + n_air, 0, sizeof(float));
        if (it < n_pulse && n_src > 0)
            aur_fdtd_inject<<<io_blocks(n_src), 256>>>(d_p2, d_si, d_sw, n_src,
                                                       pulse[it]);
        if (n_rec > 0)
            aur_fdtd_gather<<<io_blocks(n_rec), 256>>>(d_p2, d_ri, d_rw, n_rec,
                                                       d_out + (size_t)it * n_rec);
        if (have_f)
            aur_fdtd_phi<<<gf, B>>>(n_faces, d_fc, d_fe, d_fg, d_p2,
                                    d_p1, d_phiF);
        if (have_r)
            aur_fdtd_psir<<<(int)((n_rfaces + B - 1) / B), B>>>(
                n_rfaces, d_rc, d_rE, d_rG, d_p2, d_p1, d_psiR);
        float* t = d_p0; d_p0 = d_p1; d_p1 = d_p2; d_p2 = t;
        if ((it % sent_every) == 0 && it > 0) {
            const size_t stride =
                ((size_t)n_air / sent_buf.size()) + 1;
            size_t ns = ((size_t)n_air + stride - 1) / stride;
            if (ns > sent_buf.size()) ns = sent_buf.size();
            /* one strided copy: 25000 single-float copies cost ~0.25 s */
            cudaMemcpy2D(sent_buf.data(), sizeof(float), d_p1, stride * sizeof(float),
                         sizeof(float), ns, cudaMemcpyDeviceToHost);
            float mx = 0.f;
            for (size_t q = 0; q < ns; ++q) {
                float v = std::fabs(sent_buf[q]);
                if (!(v == v)) { mx = -1.f; break; }
                if (v > mx) mx = v;
            }
            if (sent_ref <= 0.f && mx > 0.f) sent_ref = mx;
            if (mx < 0.f
                || (sent_ref > 0.f && mx > 1e4f * sent_ref)) {
                std::fprintf(stderr,
                    "[aur_fdtd] instability detected at step %d "
                    "(sampled max|p| %.3e, ref %.3e)\n",
                    it, (double)mx, (double)sent_ref);
                gpu_note("fdtd: instability (sentinel)",
                         cudaErrorUnknown);
                for (void* p : owned) cudaFree(p);
                return 41;
            }
        }
        if (dbg_every > 0 && (it % dbg_every) == 0) {
            cudaDeviceSynchronize();
            const char* bad = nullptr;
            struct { const float* p; size_t n; const char* s; }
            scans[] = {
                { d_p1, (size_t)n_air + 1, "pression p" },
                { d_psi, (size_t)n_pml, "psi (CPML)" },
                { d_phiP, (size_t)n_pml, "phi (CPML)" },
                { d_phiF, have_f ? 2 * (size_t)n_faces : 0,
                  "phi (parois)" },
                { d_D, have_d ? (size_t)n_air : 0, "drive D" },
            };
            for (auto& sc : scans) {
                if (!sc.p || !sc.n) continue;
                if (dbg_buf.size() < sc.n) dbg_buf.resize(sc.n);
                cudaMemcpy(dbg_buf.data(), sc.p,
                           sc.n * sizeof(float),
                           cudaMemcpyDeviceToHost);
                for (size_t q = 0; q < sc.n; ++q)
                    if (!std::isfinite(dbg_buf[q])) {
                        bad = sc.s;
                        std::fprintf(stderr,
                            "[aur_fdtd dbg] NON-FINITE in %s at "
                            "step %d (index %zu, value %g)\n",
                            sc.s, it, q, (double)dbg_buf[q]);
                        break;
                    }
                if (bad) break;
            }
            if (bad) { gpu_note("fdtd debug: divergence",
                                cudaErrorUnknown);
                       for (void* p : owned) cudaFree(p);
                       return 40; }
        }
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) { gpu_note("fdtd kernels", e); goto fail; }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { gpu_note("fdtd sync", e); goto fail; }
    if (timing) {
        auto t_c = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[aur_fdtd] one device: %d steps %.2f s (%.3f ms per step)\n",
                     n_steps, std::chrono::duration<double>(t_c - t_b).count(),
                     1e3 * std::chrono::duration<double>(t_c - t_b).count() / n_steps);
    }
    if (n_rec > 0) {
        e = cudaMemcpy(out, d_out,
                       (size_t)n_steps * n_rec * sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { gpu_note("fdtd probe copy-back", e);
                                goto fail; }
    }
    for (void* p : owned) cudaFree(p);
    return 0;
    }
fail:
    for (void* p : owned) cudaFree(p);
    return 31;
    #undef UP
    #undef AZ
#else
    /* EMU: sequential, the native algorithm */
    std::vector<float> P0((size_t)n_air + 1, 0.f),
                       P1((size_t)n_air + 1, 0.f),
                       P2((size_t)n_air + 1, 0.f);
    float* p0 = P0.data(); float* p1 = P1.data();
    float* p2 = P2.data();
    std::vector<float> D, phiF;
    const bool have_re4 = (n_rfaces > 0 && r_cell && c_drv);
    std::vector<float> psiR;
    if (n_faces > 0 || have_re4)
        D.assign((size_t)n_air, 0.f);
    if (n_faces > 0)
        phiF.assign(2 * (size_t)n_faces, 0.f);
    if (have_re4)
        psiR.assign(2 * (size_t)n_rfaces, 0.f);
    int64_t off[4] = {0, 0, 0, 0};
    int64_t n_pml = 0;
    if (pml_counts) {
        for (int a = 0; a < 3; ++a) { off[a] = n_pml;
                                      n_pml += pml_counts[a]; }
        off[3] = n_pml;
    }
    std::vector<float> psi((size_t)n_pml, 0.f), phiP((size_t)n_pml, 0.f);
    const bool have_f = (n_faces > 0 && f_cell && c_drv);
    const bool have_d = (have_f || have_re4);
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        if (have_d)
            std::fill(D.begin(), D.end(), 0.f);
        if (have_f) {
            for (int64_t f = 0; f < n_faces; ++f) {
                const int64_t c = f_cell[f];
                D[c] += f_Asum[f] * p1[c]
                      - (f_l[2 * f] * phiF[2 * f]
                         + f_l[2 * f + 1] * phiF[2 * f + 1]);
            }
        }
        if (have_re4)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                D[c] += r_A[f] * p1[c]
                      - (r_c1[f] * psiR[2 * f]
                         + r_c2[f] * psiR[2 * f + 1]);
            }
        for (int64_t i = 0; i < n_air; ++i) {
            const int32_t* q = nb + 6 * i;
            float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                    + p1[q[3]] + p1[q[4]] + p1[q[5]];
            if (mask4[i]) {
                const int32_t* q2 = nb2 + 6 * i;
                float s2 = p1[q2[0]] + p1[q2[1]] + p1[q2[2]]
                         + p1[q2[3]] + p1[q2[4]] + p1[q2[5]];
                p2[i] = c4c * p1[i] + c4nb * s + c4nb2 * s2 - p0[i];
            } else {
                p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i];
                if (have_d) p2[i] -= c_drv[i] * D[i];
            }
        }
        for (int64_t r = 0; r < n_lrows; ++r) {
            const int64_t i = l_rows[r];
            float acc = 0.f;
            for (int64_t k = l_ptr[r]; k < l_ptr[r + 1]; ++k)
                acc += l_k[k] * (p1[l_nb[k]] - p1[i]);
            p2[i] += acc;
        }
        if (n_pml > 0) {
            for (int64_t k = 0; k < n_pml; ++k) {
                const float dh = p1[pml_nbp[k]] - p1[pml_cells[k]];
                psi[k] = pml_b[k] * psi[k] + pml_apsi[k] * dh;
            }
            for (int a = 0; a < 3; ++a) {
                const int64_t o = off[a];
                for (int64_t k = o; k < off[a + 1]; ++k) {
                    const int64_t lm = pml_locm[k];
                    const float pm = (lm >= 0) ? psi[o + lm] : 0.f;
                    const float dpsi = psi[k] - pm;
                    const int64_t cc = pml_cells[k];
                    const float d2 = p1[pml_nbp[k]] - 2.f * p1[cc]
                                   + p1[pml_nbm[k]];
                    phiP[k] = pml_b[k] * phiP[k] + pml_a[k] * (d2 + dpsi);
                    p2[cc] += lam2 * (dpsi + phiP[k]);
                }
            }
        }
        p2[n_air] = 0.f;
        if (it < n_pulse)
            for (int32_t k = 0; k < n_src; ++k)
                p2[src_idx[k]] += pulse[it] * src_w[k];
        for (int32_t r = 0; r < n_rec; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 8; ++k)
                acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
            out[(size_t)it * n_rec + r] = acc;
        }
        if (have_f) {
            for (int64_t f = 0; f < n_faces; ++f) {
                const float pc = 0.5f * (p2[f_cell[f]] + p1[f_cell[f]]);
                phiF[2 * f]     = f_e[2 * f] * phiF[2 * f]
                                + f_g[2 * f] * pc;
                phiF[2 * f + 1] = f_e[2 * f + 1] * phiF[2 * f + 1]
                                + f_g[2 * f + 1] * pc;
            }
        }
        if (have_re4)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                const float pc = 0.5f * (p2[c] + p1[c]);
                const float s1 = psiR[2 * f], s2 = psiR[2 * f + 1];
                psiR[2 * f]     = r_E[4 * f]     * s1 + r_E[4 * f + 1] * s2
                                + r_G[2 * f]     * pc;
                psiR[2 * f + 1] = r_E[4 * f + 2] * s1 + r_E[4 * f + 3] * s2
                                + r_G[2 * f + 1] * pc;
            }
        float* tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
    }
    return 0;
#endif
}


AUR_API int32_t aur_fdtd_run4(const int32_t* nb, const int32_t* nb2,
                              const uint8_t* mask4,
                              const float* c_now, const float* c_old,
                              const float* c_nb, const float* c_drv,
                              float c4c, float c4nb, float c4nb2,
                              float lam2, int64_t n_air,
                              int64_t n_faces, const int32_t* f_cell,
                              const float* f_e, const float* f_g,
                              const float* f_l, const float* f_Asum,
                              int64_t n_rfaces, const int32_t* r_cell,
                              const float* r_A, const float* r_c1,
                              const float* r_c2, const float* r_E,
                              const float* r_G,
                              const int64_t* pml_counts,
                              const int32_t* pml_cells,
                              const float* pml_b, const float* pml_a,
                              const float* pml_apsi,
                              const int32_t* pml_nbm,
                              const int32_t* pml_nbp,
                              const int32_t* pml_locm,
                              const float* pulse, int32_t n_pulse,
                              const int32_t* src_idx, const float* src_w,
                              int32_t n_src,
                              const int32_t* rec_idx, const float* rec_w,
                              int32_t n_rec,
                              int32_t n_steps, int32_t n_threads,
                              float* out) {
    return fdtd_run4_impl(nb, nb2, mask4, c_now, c_old, c_nb, c_drv, c4c, c4nb, c4nb2, lam2, n_air,
                          n_faces, f_cell, f_e, f_g, f_l, f_Asum, n_rfaces, r_cell, r_A,
                          r_c1, r_c2, r_E, r_G, pml_counts, pml_cells, pml_b, pml_a,
                          pml_apsi, pml_nbm, pml_nbp, pml_locm,
                          0, nullptr, nullptr, nullptr, nullptr,
                          pulse, n_pulse, src_idx, src_w, n_src, rec_idx, rec_w, n_rec,
                          n_steps, n_threads, out);
}

/* run4 + sparse links (conformal tangential apertures). */
AUR_API int32_t aur_fdtd_run5(const int32_t* nb, const int32_t* nb2,
                              const uint8_t* mask4,
                              const float* c_now, const float* c_old,
                              const float* c_nb, const float* c_drv,
                              float c4c, float c4nb, float c4nb2,
                              float lam2, int64_t n_air,
                              int64_t n_faces, const int32_t* f_cell,
                              const float* f_e, const float* f_g,
                              const float* f_l, const float* f_Asum,
                              int64_t n_rfaces, const int32_t* r_cell,
                              const float* r_A, const float* r_c1,
                              const float* r_c2, const float* r_E,
                              const float* r_G,
                              const int64_t* pml_counts,
                              const int32_t* pml_cells,
                              const float* pml_b, const float* pml_a,
                              const float* pml_apsi,
                              const int32_t* pml_nbm,
                              const int32_t* pml_nbp,
                              const int32_t* pml_locm,
                              int64_t n_lrows, const int32_t* l_rows,
                              const int64_t* l_ptr, const int32_t* l_nb,
                              const float* l_k,
                              const float* pulse, int32_t n_pulse,
                              const int32_t* src_idx, const float* src_w,
                              int32_t n_src,
                              const int32_t* rec_idx, const float* rec_w,
                              int32_t n_rec,
                              int32_t n_steps, int32_t n_threads,
                              float* out) {
    return fdtd_run4_impl(nb, nb2, mask4, c_now, c_old, c_nb, c_drv, c4c, c4nb, c4nb2, lam2, n_air,
                          n_faces, f_cell, f_e, f_g, f_l, f_Asum, n_rfaces, r_cell, r_A,
                          r_c1, r_c2, r_E, r_G, pml_counts, pml_cells, pml_b, pml_a,
                          pml_apsi, pml_nbm, pml_nbp, pml_locm,
                          n_lrows, l_rows, l_ptr, l_nb, l_k,
                          pulse, n_pulse, src_idx, src_w, n_src, rec_idx, rec_w, n_rec,
                          n_steps, n_threads, out);
}

} /* extern "C" */

/* run4 split over several devices. Air cells are numbered x plane after x plane, so a slab of
 * whole x planes is a contiguous range: slab k owns [g0, g1) and keeps a copy of [hl, hr), two
 * x planes each side, refreshed after every step (all the stencil reaches; cuts avoid the x PML).
 * Indices are relative to g0 (left copy negative), ghost at hr - g0. Each cell goes through the
 * operations of one domain in the same order: bit for bit against a single domain. */
namespace {

struct MSlab {
    int64_t g0 = 0, g1 = 0, hl = 0, hr = 0;
    int64_t n_own() const { return g1 - g0; }
    int64_t n_loc() const { return hr - hl; }
    int64_t ghost() const { return hr - g0; }
    int64_t L() const { return g0 - hl; }
    std::vector<int32_t> nb, nb2, fc, rc, pc, pm, pp, pl, si, ri;
    std::vector<uint8_t> m4;
    std::vector<float> cn, co, cb, cd, fe, fg, fl, fA, rA, rc1, rc2, rE, rG, pb, pa, ps, sw, rw;
    int64_t poff[4] = {0, 0, 0, 0};
    std::vector<int32_t> recs;          /* global numbers of the receivers gathered here */
    std::vector<int32_t> lr, ln;        /* tangential links (conformal_aniso), local CSR */
    std::vector<int64_t> lp;
    std::vector<float> lk;
    int64_t nf = 0, nr = 0, np = 0, nsi = 0, nlr = 0;
    void release() {                    /* host copies, once uploaded */
        std::vector<int32_t>().swap(nb); std::vector<int32_t>().swap(nb2);
        std::vector<int32_t>().swap(fc); std::vector<int32_t>().swap(rc);
        std::vector<int32_t>().swap(pc); std::vector<int32_t>().swap(pm);
        std::vector<int32_t>().swap(pp); std::vector<int32_t>().swap(pl);
        std::vector<uint8_t>().swap(m4);
        std::vector<int32_t>().swap(lr); std::vector<int32_t>().swap(ln);
        std::vector<int64_t>().swap(lp);
        for (auto* v : {&cn, &co, &cb, &cd, &fe, &fg, &fl, &fA, &rA, &rc1, &rc2, &rE, &rG,
                        &pb, &pa, &ps, &lk})
            std::vector<float>().swap(*v);
    }
};

inline bool mremap(int64_t g, const MSlab& s, int64_t n_air, int32_t* out) {
    if (g == n_air) { *out = (int32_t)s.ghost(); return true; }
    if (g < s.hl || g >= s.hr) return false;
    *out = (int32_t)(g - s.g0);
    return true;
}

struct MArgs {
    const int32_t *nb, *nb2; const uint8_t* mask4;
    const float *c_now, *c_old, *c_nb, *c_drv; int64_t n_air;
    int64_t n_faces; const int32_t* f_cell; const float *f_e, *f_g, *f_l, *f_Asum;
    int64_t n_rfaces; const int32_t* r_cell; const float *r_A, *r_c1, *r_c2, *r_E, *r_G;
    const int64_t* pml_counts; const int32_t* pml_cells; const float *pml_b, *pml_a, *pml_apsi;
    const int32_t *pml_nbm, *pml_nbp, *pml_locm;
    int64_t n_lrows = 0; const int32_t* l_rows = nullptr; const int64_t* l_ptr = nullptr;
    const int32_t* l_nb = nullptr; const float* l_k = nullptr;
    /* slab sub grid (aur_fdtd_slabs_add): index j is virtual j + shift, its ghost sub_n is
     * the virtual ghost n_air; sub_n = -1: the arrays ARE the global grid */
    int64_t shift = 0, sub_n = -1;
};

/* AUR_FDTD_SLAB_DEBUG=1: which reference left the copy of a slab (code 42) */
inline void slab_debug(const char* what, const MSlab& s, int64_t g = -1) {
    if (std::getenv("AUR_FDTD_SLAB_DEBUG"))
        std::fprintf(stderr, "[aur_fdtd] slab [%lld, %lld) copy [%lld, %lld): %s %lld outside\n",
                     (long long)s.g0, (long long)s.g1, (long long)s.hl, (long long)s.hr, what,
                     (long long)g);
}

inline int64_t vix(const MArgs& A, int64_t j) {
    return (A.sub_n >= 0 && j == A.sub_n) ? A.n_air : j + A.shift;
}

/* Slab bounds. 0, or 42 on bad cuts. */
int32_t plan_bounds(int32_t n_dev, const int64_t* cuts, const int64_t* halo, int64_t n_air,
                    std::vector<MSlab>& S) {
    S.assign((size_t)n_dev, MSlab());
    for (int32_t k = 0; k < n_dev; ++k) {
        MSlab& s = S[(size_t)k];
        s.g0 = cuts[k]; s.g1 = cuts[k + 1]; s.hl = halo[2 * k]; s.hr = halo[2 * k + 1];
        if (!(0 <= s.hl && s.hl <= s.g0 && s.g0 < s.g1 && s.g1 <= s.hr && s.hr <= n_air))
            return 42;
        if (k > 0 && (s.hl < S[(size_t)k - 1].g0 || S[(size_t)k - 1].g1 != s.g0)) return 42;
    }
    for (int32_t k = 0; k + 1 < n_dev; ++k)          /* the right copy comes from owned cells */
        if (S[(size_t)k].hr > S[(size_t)k + 1].g1) return 42;
    return 0;
}

/* A source tap goes to the slab owning its cell, a receiver to the slab owning its first tap
 * (its 8 taps must lie in that slab's copy). 0, or 42. */
template <typename IX>
int32_t assign_points(std::vector<MSlab>& S, int64_t n_air, const IX* src_idx,
                      const float* src_w, int32_t n_src, const IX* rec_idx,
                      const float* rec_w, int32_t n_rec) {
    const int32_t n_dev = (int32_t)S.size();
    for (MSlab& s : S) {
        s.si.clear(); s.sw.clear(); s.ri.clear(); s.rw.clear(); s.recs.clear();
        for (int32_t q = 0; q < n_src; ++q) {
            const int64_t g = src_idx[q];
            if (g < s.g0 || g >= s.g1) continue;
            s.si.push_back((int32_t)(g - s.g0));
            s.sw.push_back(src_w[q]);
        }
        s.nsi = (int64_t)s.si.size();
    }
    for (int32_t r = 0; r < n_rec; ++r) {
        int64_t first = -1;
        for (int q = 0; q < 8; ++q)
            if ((int64_t)rec_idx[8 * r + q] != n_air) { first = rec_idx[8 * r + q]; break; }
        int32_t k = 0;
        if (first >= 0)
            while (k + 1 < n_dev && first >= S[(size_t)k + 1].g0) ++k;
        MSlab& s = S[(size_t)k];
        for (int q = 0; q < 8; ++q) {
            int32_t v;
            if (!mremap(rec_idx[8 * r + q], s, n_air, &v)) return 42;
            s.ri.push_back(v);
            s.rw.push_back(rec_w[8 * r + q]);
        }
        s.recs.push_back(r);
    }
    return 0;
}

/* The arrays of slab s (global order kept, indices made local). 0, or 42 when a reference
 * leaves the slab copy (a cut through a PML layer, a slab thinner than its neighbour's halo). */
int32_t fill_mslab(MSlab& s, const MArgs& A) {
    const int64_t n_air = A.n_air;
    int64_t off[4] = {0, 0, 0, 0};
    if (A.pml_counts)
        for (int a = 0; a < 3; ++a) off[a + 1] = off[a] + A.pml_counts[a];
    const int64_t n = s.n_own();
    const int64_t p0 = s.g0 - A.shift;          /* position of the slab's first cell in A */
    s.nb.resize((size_t)(6 * n));
    s.nb2.resize((size_t)(6 * n));
    {   /* neighbour remap, the bulk of the work: threaded */
        unsigned hc = std::thread::hardware_concurrency();
        const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(hc ? hc : 4, 32));
        const int64_t chunk = (n + nt - 1) / nt;
        std::atomic<int> bad(0);
        std::vector<std::thread> pool;
        for (int64_t t = 0; t < nt; ++t)
            pool.emplace_back([&, t]() {
                for (int64_t i = t * chunk; i < std::min(n, (t + 1) * chunk); ++i)
                    for (int f = 0; f < 6; ++f) {
                        if (!mremap(vix(A, A.nb[6 * (p0 + i) + f]), s, n_air,
                                    &s.nb[(size_t)(6 * i + f)])
                            || !mremap(vix(A, A.nb2[6 * (p0 + i) + f]), s, n_air,
                                       &s.nb2[(size_t)(6 * i + f)])) {
                            bad.store(1);
                            return;
                        }
                    }
            });
        for (auto& th : pool) th.join();
        if (bad.load()) { slab_debug("neighbour", s); return 42; }
    }
    s.m4.assign(A.mask4 + p0, A.mask4 + p0 + n);
    s.cn.assign(A.c_now + p0, A.c_now + p0 + n);
    s.co.assign(A.c_old + p0, A.c_old + p0 + n);
    s.cb.assign(A.c_nb + p0, A.c_nb + p0 + n);
    if (A.c_drv) s.cd.assign(A.c_drv + p0, A.c_drv + p0 + n);
    for (int64_t f = 0; f < A.n_faces; ++f) {
        const int64_t c = vix(A, A.f_cell[f]);
        if (c < s.g0 || c >= s.g1) continue;
        s.fc.push_back((int32_t)(c - s.g0));
        s.fe.push_back(A.f_e[2 * f]); s.fe.push_back(A.f_e[2 * f + 1]);
        s.fg.push_back(A.f_g[2 * f]); s.fg.push_back(A.f_g[2 * f + 1]);
        s.fl.push_back(A.f_l[2 * f]); s.fl.push_back(A.f_l[2 * f + 1]);
        s.fA.push_back(A.f_Asum[f]);
    }
    for (int64_t f = 0; f < A.n_rfaces; ++f) {
        const int64_t c = vix(A, A.r_cell[f]);
        if (c < s.g0 || c >= s.g1) continue;
        s.rc.push_back((int32_t)(c - s.g0));
        s.rA.push_back(A.r_A[f]); s.rc1.push_back(A.r_c1[f]); s.rc2.push_back(A.r_c2[f]);
        for (int q = 0; q < 4; ++q) s.rE.push_back(A.r_E[4 * f + q]);
        s.rG.push_back(A.r_G[2 * f]); s.rG.push_back(A.r_G[2 * f + 1]);
    }
    for (int a = 0; a < 3; ++a) {
        s.poff[a] = (int64_t)s.pc.size();
        const int64_t cnt = off[a + 1] - off[a];
        /* an axis table is sorted by cell: the slab's entries are the contiguous range [lo, hi) */
        const int32_t* cb = A.pml_cells + off[a];
        const int64_t lo = std::lower_bound(cb, cb + cnt, (int32_t)p0) - cb;
        const int64_t hi = std::lower_bound(cb, cb + cnt, (int32_t)(p0 + n)) - cb;
        for (int64_t j = lo; j < hi; ++j) {
            const int64_t g = off[a] + j;
            int32_t v;
            s.pc.push_back((int32_t)(vix(A, A.pml_cells[g]) - s.g0));
            if (!mremap(vix(A, A.pml_nbm[g]), s, n_air, &v)) { slab_debug("pml nbm", s); return 42; }
            s.pm.push_back(v);
            if (!mremap(vix(A, A.pml_nbp[g]), s, n_air, &v)) { slab_debug("pml nbp", s); return 42; }
            s.pp.push_back(v);
            const int64_t lm = A.pml_locm[g];
            if (lm >= 0 && (lm < lo || lm >= hi)) { slab_debug("pml locm", s); return 42; }
            s.pl.push_back(lm >= 0 ? (int32_t)(lm - lo) : -1);
            s.pb.push_back(A.pml_b[g]); s.pa.push_back(A.pml_a[g]); s.ps.push_back(A.pml_apsi[g]);
        }
    }
    s.poff[3] = (int64_t)s.pc.size();
    /* tangential links: rows of the owned cells in global order, neighbours inside the copy */
    s.lr.clear(); s.ln.clear(); s.lk.clear(); s.lp.assign(1, 0);
    for (int64_t r = 0; r < A.n_lrows; ++r) {
        const int64_t c = vix(A, A.l_rows[r]);
        if (c < s.g0 || c >= s.g1) continue;
        s.lr.push_back((int32_t)(c - s.g0));
        for (int64_t q = A.l_ptr[r]; q < A.l_ptr[r + 1]; ++q) {
            int32_t v;
            if (!mremap(vix(A, A.l_nb[q]), s, n_air, &v)) { slab_debug("link", s, vix(A, A.l_nb[q])); return 42; }
            s.ln.push_back(v);
            s.lk.push_back(A.l_k[q]);
        }
        s.lp.push_back((int64_t)s.ln.size());
    }
    s.nf = (int64_t)s.fc.size(); s.nr = (int64_t)s.rc.size();
    s.np = (int64_t)s.pc.size(); s.nlr = (int64_t)s.lr.size();
    return 0;
}

/* EMU multi-slab run: the single-domain loop slab by slab, halo copies between update and gather. */
[[maybe_unused]] int32_t run4m_host(std::vector<MSlab>& S, bool have_f, bool have_r, float c4c,
                                    float c4nb, float c4nb2, float lam2, const float* pulse,
                                    int32_t n_pulse, int32_t n_rec, int32_t n_steps, float* out) {
    const size_t K = S.size();
    const bool have_d = have_f || have_r;
    std::vector<std::vector<float>> B0(K), B1(K), B2(K), D(K), phiF(K), psiR(K), psi(K), phiP(K);
    std::vector<float*> p0(K), p1(K), p2(K);
    for (size_t k = 0; k < K; ++k) {
        const MSlab& s = S[k];
        B0[k].assign((size_t)s.n_loc() + 1, 0.f);
        B1[k].assign((size_t)s.n_loc() + 1, 0.f);
        B2[k].assign((size_t)s.n_loc() + 1, 0.f);
        p0[k] = B0[k].data() + s.L(); p1[k] = B1[k].data() + s.L(); p2[k] = B2[k].data() + s.L();
        if (have_d) D[k].assign((size_t)s.n_own(), 0.f);
        phiF[k].assign(s.fe.size(), 0.f);
        psiR[k].assign(2 * s.rc.size(), 0.f);
        psi[k].assign(s.pc.size(), 0.f);
        phiP[k].assign(s.pc.size(), 0.f);
    }
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        for (size_t k = 0; k < K; ++k) {
            const MSlab& s = S[k];
            float* P0 = p0[k]; float* P1 = p1[k]; float* P2 = p2[k];
            if (have_d) std::fill(D[k].begin(), D[k].end(), 0.f);
            if (have_f)
                for (size_t f = 0; f < s.fc.size(); ++f) {
                    const int64_t c = s.fc[f];
                    D[k][(size_t)c] += s.fA[f] * P1[c]
                        - (s.fl[2 * f] * phiF[k][2 * f] + s.fl[2 * f + 1] * phiF[k][2 * f + 1]);
                }
            if (have_r)
                for (size_t f = 0; f < s.rc.size(); ++f) {
                    const int64_t c = s.rc[f];
                    D[k][(size_t)c] += s.rA[f] * P1[c]
                        - (s.rc1[f] * psiR[k][2 * f] + s.rc2[f] * psiR[k][2 * f + 1]);
                }
            for (int64_t i = 0; i < s.n_own(); ++i) {
                const int32_t* q = &s.nb[(size_t)(6 * i)];
                float sm = P1[q[0]] + P1[q[1]] + P1[q[2]] + P1[q[3]] + P1[q[4]] + P1[q[5]];
                if (s.m4[(size_t)i]) {
                    const int32_t* q2 = &s.nb2[(size_t)(6 * i)];
                    float s2 = P1[q2[0]] + P1[q2[1]] + P1[q2[2]]
                             + P1[q2[3]] + P1[q2[4]] + P1[q2[5]];
                    P2[i] = c4c * P1[i] + c4nb * sm + c4nb2 * s2 - P0[i];
                } else {
                    P2[i] = s.cn[(size_t)i] * P1[i] + s.cb[(size_t)i] * sm
                          + s.co[(size_t)i] * P0[i];
                    if (have_d) P2[i] -= s.cd[(size_t)i] * D[k][(size_t)i];
                }
            }
            for (int64_t r = 0; r < s.nlr; ++r) {
                const int64_t i = s.lr[(size_t)r];
                float acc = 0.f;
                for (int64_t q = s.lp[(size_t)r]; q < s.lp[(size_t)r + 1]; ++q)
                    acc += s.lk[(size_t)q] * (P1[s.ln[(size_t)q]] - P1[i]);
                P2[i] += acc;
            }
            for (size_t j = 0; j < s.pc.size(); ++j) {
                const float dh = P1[s.pp[j]] - P1[s.pc[j]];
                psi[k][j] = s.pb[j] * psi[k][j] + s.ps[j] * dh;
            }
            for (int a = 0; a < 3; ++a) {
                const int64_t o = s.poff[a];
                for (int64_t j = o; j < s.poff[a + 1]; ++j) {
                    const int64_t lm = s.pl[(size_t)j];
                    const float pm = (lm >= 0) ? psi[k][(size_t)(o + lm)] : 0.f;
                    const float dpsi = psi[k][(size_t)j] - pm;
                    const int64_t cc = s.pc[(size_t)j];
                    const float d2 = P1[s.pp[(size_t)j]] - 2.f * P1[cc] + P1[s.pm[(size_t)j]];
                    phiP[k][(size_t)j] = s.pb[(size_t)j] * phiP[k][(size_t)j]
                                       + s.pa[(size_t)j] * (d2 + dpsi);
                    P2[cc] += lam2 * (dpsi + phiP[k][(size_t)j]);
                }
            }
            P2[s.ghost()] = 0.f;
            if (it < n_pulse)
                for (size_t q = 0; q < s.si.size(); ++q) P2[s.si[q]] += pulse[it] * s.sw[q];
        }
        for (size_t k = 0; k + 1 < K; ++k) {
            const MSlab& a = S[k];
            const MSlab& b = S[k + 1];
            std::memcpy(p2[k + 1] + (b.hl - b.g0), p2[k] + (b.hl - a.g0),
                        (size_t)(b.g0 - b.hl) * sizeof(float));
            std::memcpy(p2[k] + (a.g1 - a.g0), p2[k + 1] + (a.g1 - b.g0),
                        (size_t)(a.hr - a.g1) * sizeof(float));
        }
        for (size_t k = 0; k < K; ++k) {
            const MSlab& s = S[k];
            float* P1 = p1[k]; float* P2 = p2[k];
            for (size_t r = 0; r < s.recs.size(); ++r) {
                float acc = 0.f;
                for (int q = 0; q < 8; ++q) acc += P2[s.ri[8 * r + q]] * s.rw[8 * r + q];
                out[(size_t)it * n_rec + s.recs[r]] = acc;
            }
            if (have_f)
                for (size_t f = 0; f < s.fc.size(); ++f) {
                    const float pc = 0.5f * (P2[s.fc[f]] + P1[s.fc[f]]);
                    phiF[k][2 * f]     = s.fe[2 * f] * phiF[k][2 * f] + s.fg[2 * f] * pc;
                    phiF[k][2 * f + 1] = s.fe[2 * f + 1] * phiF[k][2 * f + 1]
                                       + s.fg[2 * f + 1] * pc;
                }
            if (have_r)
                for (size_t f = 0; f < s.rc.size(); ++f) {
                    const int64_t c = s.rc[f];
                    const float pc = 0.5f * (P2[c] + P1[c]);
                    const float s1 = psiR[k][2 * f], s2 = psiR[k][2 * f + 1];
                    psiR[k][2 * f]     = s.rE[4 * f] * s1 + s.rE[4 * f + 1] * s2 + s.rG[2 * f] * pc;
                    psiR[k][2 * f + 1] = s.rE[4 * f + 2] * s1 + s.rE[4 * f + 3] * s2
                                       + s.rG[2 * f + 1] * pc;
                }
            float* t = p0[k]; p0[k] = p1[k]; p1[k] = p2[k]; p2[k] = t;
        }
    }
    return 0;
}

#if AUR_GPU
GLOBALFN void aur_fdtd_gather_n(const float* p2, const int32_t* rec_idx, const float* rec_w,
                                int32_t n_rec, float* out_step) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rec) return;
    float acc = 0.f;
    for (int k = 0; k < 8; ++k)
        acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
    out_step[r] = acc;
}

struct MDev {
    int dev = 0;
    std::vector<void*> owned;
    int32_t *nb = nullptr, *nb2 = nullptr, *fc = nullptr, *rc = nullptr, *pc = nullptr,
            *pm = nullptr, *pp = nullptr, *pl = nullptr, *si = nullptr, *ri = nullptr;
    uint8_t* m4 = nullptr;
    int32_t *lr = nullptr, *ln = nullptr;
    int64_t* lp = nullptr;
    float* lk = nullptr;
    int32_t *wc = nullptr, *wfi = nullptr, *wri = nullptr;   /* wall faces by cell (DriveCsr) */
    int64_t *wfp = nullptr, *wrp = nullptr;
    int64_t nwc = 0;
    float *cn = nullptr, *co = nullptr, *cb = nullptr, *cd = nullptr, *fe = nullptr,
          *fg = nullptr, *fl = nullptr, *fA = nullptr, *rA = nullptr, *rc1 = nullptr,
          *rc2 = nullptr, *rE = nullptr, *rG = nullptr, *pb = nullptr, *pa = nullptr,
          *ps = nullptr, *sw = nullptr, *rw = nullptr;
    float *P0 = nullptr, *P1 = nullptr, *P2 = nullptr;
    float *D = nullptr, *phiF = nullptr, *psiR = nullptr, *psi = nullptr, *phiP = nullptr,
          *out = nullptr;
    size_t nl = 0, nD = 0, nF = 0, nR = 0, nP = 0;   /* state sizes, reset before each run */
    std::vector<void*> run_owned;                    /* points and probes of the current run */
};

template <typename T>
bool m_up(MDev& d, T** dst, const std::vector<T>& v, const char* what) {
    *dst = nullptr;
    if (v.empty()) return true;
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, v.size() * sizeof(T));
    if (e != cudaSuccess) { gpu_note(what, e); return false; }
    d.owned.push_back(p);
    e = cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { gpu_note(what, e); return false; }
    *dst = (T*)p;
    return true;
}

bool m_zero(MDev& d, float** dst, size_t n, const char* what) {
    *dst = nullptr;
    if (n == 0) return true;
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, n * sizeof(float));
    if (e != cudaSuccess) { gpu_note(what, e); return false; }
    d.owned.push_back(p);
    e = cudaMemset(p, 0, n * sizeof(float));
    if (e != cudaSuccess) { gpu_note(what, e); return false; }
    *dst = (float*)p;
    return true;
}

/* A grid kept on its devices between runs; each run starts from zero states. */
struct MSession {
    std::vector<MSlab> S;
    std::vector<int32_t> devs;
    int64_t n_air = 0;
    bool have_f = false, have_r = false;
    float c4c = 0.f, c4nb = 0.f, c4nb2 = 0.f, lam2 = 0.f;
    std::vector<MDev> M;
};

int32_t upload_mslab(MSlab& s, MDev& d, bool have_d);

int32_t sess_create_gpu(MSession& X, const MArgs& A) {
    const size_t K = X.S.size();
    const bool have_d = X.have_f || X.have_r;
    X.M.assign(K, MDev());
    cudaError_t e;
    for (size_t k = 0; k < K; ++k)
        for (int j : {(int)k - 1, (int)k + 1}) {
            if (j < 0 || j >= (int)K || X.devs[j] == X.devs[k]) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, X.devs[k], X.devs[j]);
            if (!can) continue;
            cudaSetDevice(X.devs[k]);
            if (cudaDeviceEnablePeerAccess(X.devs[j], 0) != cudaSuccess) cudaGetLastError();
        }
    /* the slabs are filled and uploaded in parallel, one host thread per device */
    auto one = [&](size_t k) -> int32_t {
        cudaError_t e;
        MSlab& s = X.S[k];
        MDev& d = X.M[k];
        d.dev = X.devs[k];
        if ((e = cudaSetDevice(d.dev)) != cudaSuccess) { gpu_note("fdtd multi set device", e); return 31; }
        if (int32_t rc = fill_mslab(s, A)) return rc;
        return upload_mslab(s, d, have_d);
    };
    std::vector<int32_t> rcs(K, 0);
    if (K == 1) {
        rcs[0] = one(0);
    } else {
        std::vector<std::thread> pool;
        for (size_t k = 0; k < K; ++k) pool.emplace_back([&, k]() { rcs[k] = one(k); });
        for (auto& th : pool) th.join();
    }
    for (int32_t rc : rcs)
        if (rc) return rc;
    return 0;
}

/* Peer access between neighbouring devices of a session. */
void sess_peers(MSession& X) {
    const size_t K = X.S.size();
    for (size_t k = 0; k < K; ++k)
        for (int j : {(int)k - 1, (int)k + 1}) {
            if (j < 0 || j >= (int)K || X.devs[j] == X.devs[k]) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, X.devs[k], X.devs[j]);
            if (!can) continue;
            cudaSetDevice(X.devs[k]);
            if (cudaDeviceEnablePeerAccess(X.devs[j], 0) != cudaSuccess) cudaGetLastError();
        }
}

/* The device arrays of a filled slab, then its host copies released. 0, or 31. */
int32_t upload_mslab(MSlab& s, MDev& d, bool have_d) {
        d.nl = (size_t)s.n_loc() + 1;
        d.nD = have_d ? (size_t)s.n_own() : 0;
        d.nF = s.fe.size(); d.nR = 2 * s.rc.size(); d.nP = s.pc.size();
        DriveCsr W;
        drive_csr((int64_t)s.fc.size(), s.fc.data(), (int64_t)s.rc.size(), s.rc.data(), W);
        d.nwc = W.n();
        bool ok = m_up(d, &d.wc, W.cell, "multi wc") && m_up(d, &d.wfp, W.fp, "multi wfp")
            && m_up(d, &d.wfi, W.fi, "multi wfi") && m_up(d, &d.wrp, W.rp, "multi wrp")
            && m_up(d, &d.wri, W.ri, "multi wri")
            && m_up(d, &d.nb, s.nb, "multi nb") && m_up(d, &d.nb2, s.nb2, "multi nb2")
            && m_up(d, &d.m4, s.m4, "multi m4") && m_up(d, &d.cn, s.cn, "multi cn")
            && m_up(d, &d.co, s.co, "multi co") && m_up(d, &d.cb, s.cb, "multi cb")
            && m_up(d, &d.cd, s.cd, "multi cd") && m_up(d, &d.fc, s.fc, "multi fc")
            && m_up(d, &d.fe, s.fe, "multi fe") && m_up(d, &d.fg, s.fg, "multi fg")
            && m_up(d, &d.fl, s.fl, "multi fl") && m_up(d, &d.fA, s.fA, "multi fA")
            && m_up(d, &d.rc, s.rc, "multi rc") && m_up(d, &d.rA, s.rA, "multi rA")
            && m_up(d, &d.rc1, s.rc1, "multi rc1") && m_up(d, &d.rc2, s.rc2, "multi rc2")
            && m_up(d, &d.rE, s.rE, "multi rE") && m_up(d, &d.rG, s.rG, "multi rG")
            && m_up(d, &d.pc, s.pc, "multi pc") && m_up(d, &d.pm, s.pm, "multi pm")
            && m_up(d, &d.pp, s.pp, "multi pp") && m_up(d, &d.pl, s.pl, "multi pl")
            && m_up(d, &d.pb, s.pb, "multi pb") && m_up(d, &d.pa, s.pa, "multi pa")
            && m_up(d, &d.ps, s.ps, "multi ps")
            && (s.nlr == 0 || (m_up(d, &d.lr, s.lr, "multi lr") && m_up(d, &d.lp, s.lp, "multi lp")
                               && m_up(d, &d.ln, s.ln, "multi ln") && m_up(d, &d.lk, s.lk, "multi lk")))
            && m_zero(d, &d.P0, d.nl, "multi p0") && m_zero(d, &d.P1, d.nl, "multi p1")
            && m_zero(d, &d.P2, d.nl, "multi p2") && m_zero(d, &d.D, d.nD, "multi D")
            && m_zero(d, &d.phiF, d.nF, "multi phiF") && m_zero(d, &d.psiR, d.nR, "multi psiR")
            && m_zero(d, &d.psi, d.nP, "multi psi") && m_zero(d, &d.phiP, d.nP, "multi phiP");
        if (!ok) return 31;
        s.release();
        return 0;
}

void sess_free_run(MSession& X) {
    for (MDev& d : X.M) {
        cudaSetDevice(d.dev);
        for (void* p : d.run_owned) cudaFree(p);
        d.run_owned.clear();
        d.si = nullptr; d.sw = nullptr; d.ri = nullptr; d.rw = nullptr; d.out = nullptr;
    }
}

void sess_destroy_gpu(MSession& X) {
    sess_free_run(X);
    for (MDev& d : X.M) {
        cudaSetDevice(d.dev);
        for (void* p : d.owned) cudaFree(p);
        d.owned.clear();
    }
}

int32_t sess_run_gpu(MSession& X, const float* pulse, int32_t n_pulse, int32_t n_rec,
                     int32_t n_steps, float* out) {
    std::vector<MSlab>& S = X.S;
    std::vector<MDev>& M = X.M;
    const size_t K = S.size();
    const bool have_f = X.have_f, have_r = X.have_r, have_d = have_f || have_r;
    const float c4c = X.c4c, c4nb = X.c4nb, c4nb2 = X.c4nb2, lam2 = X.lam2;
    int32_t rc = 0;
    cudaError_t e;
    const bool timing = std::getenv("AUR_FDTD_TIMING") != nullptr;
    auto t_a = std::chrono::steady_clock::now();
    for (size_t k = 0; k < K && rc == 0; ++k) {
        const MSlab& s = S[k];
        MDev& d = M[k];
        cudaSetDevice(d.dev);
        for (float* z : {d.P0, d.P1, d.P2})
            if (cudaMemset(z, 0, d.nl * sizeof(float)) != cudaSuccess) rc = 31;
        if (d.nD) cudaMemset(d.D, 0, d.nD * sizeof(float));
        if (d.nF) cudaMemset(d.phiF, 0, d.nF * sizeof(float));
        if (d.nR) cudaMemset(d.psiR, 0, d.nR * sizeof(float));
        if (d.nP) { cudaMemset(d.psi, 0, d.nP * sizeof(float));
                    cudaMemset(d.phiP, 0, d.nP * sizeof(float)); }
        std::vector<void*> keep;
        std::swap(keep, d.owned);                    /* m_up/m_zero record into owned */
        bool ok = m_up(d, &d.si, s.si, "multi si") && m_up(d, &d.sw, s.sw, "multi sw")
            && m_up(d, &d.ri, s.ri, "multi ri") && m_up(d, &d.rw, s.rw, "multi rw")
            && m_zero(d, &d.out, (size_t)n_steps * (s.recs.empty() ? 1 : s.recs.size()),
                      "multi out");
        d.run_owned.insert(d.run_owned.end(), d.owned.begin(), d.owned.end());
        std::swap(keep, d.owned);
        if (!ok) rc = 31;
    }
    auto t_b = std::chrono::steady_clock::now();
    const int B = 256;
    /* No host sync per step: one stream per device, cross-device order through events. Step n
     * on device k: update (drive, stencil, PML, ghost, source), record upd[k]; wait both
     * neighbours' upd, copy the edge planes into their halo copies (their step-n update never
     * touches them), record cpy[k]; wait both neighbours' cpy, then gather, walls, sections.
     * The host syncs every 512 steps for the instability sentinel. */
    std::vector<cudaStream_t> st(K, nullptr);
    std::vector<cudaEvent_t> upd(K, nullptr), cpy(K, nullptr);
    for (size_t k = 0; k < K && rc == 0; ++k) {
        cudaSetDevice(M[k].dev);
        if (cudaStreamCreateWithFlags(&st[k], cudaStreamNonBlocking) != cudaSuccess
            || cudaEventCreateWithFlags(&upd[k], cudaEventDisableTiming) != cudaSuccess
            || cudaEventCreateWithFlags(&cpy[k], cudaEventDisableTiming) != cudaSuccess) {
            gpu_note("fdtd multi streams", cudaGetLastError());
            rc = 31;
        }
    }
    const int sent_every = 512;
    float sent_ref = 0.f;
    std::vector<std::vector<float>> sbuf(K);
    std::vector<int64_t> sstride(K, 1);
    for (size_t k = 0; k < K; ++k) {
        const int64_t ns = 25000 / (int64_t)K + 1;
        sstride[k] = S[k].n_own() / ns + 1;
        sbuf[k].resize((size_t)((S[k].n_own() + sstride[k] - 1) / sstride[k]));
    }
    for (int32_t it = 0; it < n_steps && rc == 0; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        for (size_t k = 0; k < K; ++k) {
            const MSlab& s = S[k];
            MDev& d = M[k];
            cudaStream_t q = st[k];
            cudaSetDevice(d.dev);
            float* p0 = d.P0 + s.L(); float* p1 = d.P1 + s.L(); float* p2 = d.P2 + s.L();
            const int64_t np = s.np;
            if (have_d && d.nwc > 0)    /* D of the other cells stays 0 (zeroed per run) */
                aur_fdtd_drive_csr<<<(int)((d.nwc + B - 1) / B), B, 0, q>>>(
                    d.nwc, d.wc, d.wfp, d.wfi, d.wrp, d.wri, d.fl, d.fA, d.phiF, d.rA, d.rc1,
                    d.rc2, d.psiR, p1, d.D);
            aur_fdtd_step4<<<(int)((s.n_own() + B - 1) / B), B, 0, q>>>(
                d.nb, d.nb2, d.m4, d.cn, d.co, d.cb, d.cd, c4c, c4nb, c4nb2, s.n_own(),
                have_d ? 1 : 0, d.D, p0, p1, p2);
            if (s.nlr > 0)
                aur_fdtd_links<<<(int)((s.nlr + B - 1) / B), B, 0, q>>>(
                    s.nlr, d.lr, d.lp, d.ln, d.lk, p1, p2);
            if (np > 0) {
                aur_fdtd_cpml_psi<<<(int)((np + B - 1) / B), B, 0, q>>>(np, d.pc, d.pp, d.pb, d.ps, p1, d.psi);
                for (int a = 0; a < 3; ++a) {
                    const int64_t cnt = s.poff[a + 1] - s.poff[a];
                    if (cnt <= 0) continue;
                    aur_fdtd_cpml_apply<<<(int)((cnt + B - 1) / B), B, 0, q>>>(
                        s.poff[a], cnt, d.pc, d.pm, d.pp, d.pl, d.pb, d.pa, d.psi, d.phiP, lam2,
                        p1, p2);
                }
            }
            cudaMemsetAsync(p2 + s.ghost(), 0, sizeof(float), q);
            if (it < n_pulse && s.nsi > 0)
                aur_fdtd_inject<<<io_blocks(s.nsi), 256, 0, q>>>(p2, d.si, d.sw, (int32_t)s.nsi,
                                                                 pulse[it]);
            cudaEventRecord(upd[k], q);
        }
        for (size_t k = 0; k < K; ++k) {
            const MSlab& s = S[k];
            MDev& d = M[k];
            cudaStream_t q = st[k];
            cudaSetDevice(d.dev);
            if (k > 0) {
                const MSlab& a = S[k - 1];
                cudaStreamWaitEvent(q, upd[k - 1], 0);
                cudaMemcpyPeerAsync(M[k - 1].P2 + a.L() + (a.g1 - a.g0), M[k - 1].dev,
                                    d.P2 + s.L() + (a.g1 - s.g0), d.dev,
                                    (size_t)(a.hr - a.g1) * sizeof(float), q);
            }
            if (k + 1 < K) {
                const MSlab& b = S[k + 1];
                cudaStreamWaitEvent(q, upd[k + 1], 0);
                cudaMemcpyPeerAsync(M[k + 1].P2 + b.L() + (b.hl - b.g0), M[k + 1].dev,
                                    d.P2 + s.L() + (b.hl - s.g0), d.dev,
                                    (size_t)(b.g0 - b.hl) * sizeof(float), q);
            }
            cudaEventRecord(cpy[k], q);
        }
        for (size_t k = 0; k < K; ++k) {
            const MSlab& s = S[k];
            MDev& d = M[k];
            cudaStream_t q = st[k];
            cudaSetDevice(d.dev);
            if (k > 0) cudaStreamWaitEvent(q, cpy[k - 1], 0);
            if (k + 1 < K) cudaStreamWaitEvent(q, cpy[k + 1], 0);
            float* p1 = d.P1 + s.L(); float* p2 = d.P2 + s.L();
            const int64_t nf = s.nf, nr = s.nr;
            const int32_t nrk = (int32_t)s.recs.size();
            if (nrk > 0)
                aur_fdtd_gather_n<<<(nrk + B - 1) / B, B, 0, q>>>(p2, d.ri, d.rw, nrk,
                                                                  d.out + (size_t)it * nrk);
            if (have_f && nf > 0)
                aur_fdtd_phi<<<(int)((nf + B - 1) / B), B, 0, q>>>(nf, d.fc, d.fe, d.fg, p2, p1, d.phiF);
            if (have_r && nr > 0)
                aur_fdtd_psir<<<(int)((nr + B - 1) / B), B, 0, q>>>(nr, d.rc, d.rE, d.rG, p2, p1, d.psiR);
            float* t = d.P0; d.P0 = d.P1; d.P1 = d.P2; d.P2 = t;
        }
        if ((it % sent_every) == 0 && it > 0) {
            float mx = 0.f;
            for (size_t k = 0; k < K && mx >= 0.f; ++k) {
                const MSlab& s = S[k];
                cudaSetDevice(M[k].dev);
                cudaError_t ee = cudaStreamSynchronize(st[k]);
                if (ee != cudaSuccess) { gpu_note("fdtd multi step", ee); rc = 31; break; }
                if (sbuf[k].empty()) continue;
                cudaMemcpy2D(sbuf[k].data(), sizeof(float), M[k].P1 + s.L(),
                             (size_t)sstride[k] * sizeof(float), sizeof(float), sbuf[k].size(),
                             cudaMemcpyDeviceToHost);
                for (float v : sbuf[k]) {
                    v = std::fabs(v);
                    if (!(v == v)) { mx = -1.f; break; }
                    if (v > mx) mx = v;
                }
            }
            if (rc) break;
            if (sent_ref <= 0.f && mx > 0.f) sent_ref = mx;
            if (mx < 0.f || (sent_ref > 0.f && mx > 1e4f * sent_ref)) {
                std::fprintf(stderr, "[aur_fdtd] instability detected at step %d (multi-device, "
                             "sampled max|p| %.3e, ref %.3e)\n", it, (double)mx, (double)sent_ref);
                gpu_note("fdtd: instability (sentinel)", cudaErrorUnknown);
                rc = 41;
            }
        }
    }
    for (size_t k = 0; k < K; ++k) {
        if (!st[k]) continue;
        cudaSetDevice(M[k].dev);
        cudaError_t ee = cudaStreamSynchronize(st[k]);
        if (timing && k + 1 == K) {
            auto t_c = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[aur_fdtd] multi %zu devices: setup %.2f s, %d steps %.2f s "
                         "(%.3f ms per step)\n", K,
                         std::chrono::duration<double>(t_b - t_a).count(), n_steps,
                         std::chrono::duration<double>(t_c - t_b).count(),
                         1e3 * std::chrono::duration<double>(t_c - t_b).count() / n_steps);
        }
        if (ee != cudaSuccess && rc == 0) { gpu_note("fdtd multi final sync", ee); rc = 31; }
        cudaStreamDestroy(st[k]);
        if (upd[k]) cudaEventDestroy(upd[k]);
        if (cpy[k]) cudaEventDestroy(cpy[k]);
    }
    auto sync_all = [&]() -> bool {
        for (size_t k = 0; k < K; ++k) {
            cudaSetDevice(M[k].dev);
            cudaError_t ee = cudaDeviceSynchronize();
            if (ee != cudaSuccess) { gpu_note("fdtd multi sync", ee); return false; }
        }
        return true;
    };
    if (rc == 0 && !sync_all()) rc = 31;
    for (size_t k = 0; k < K && rc == 0; ++k) {
        const MSlab& s = S[k];
        if (s.recs.empty()) continue;
        std::vector<float> o((size_t)n_steps * s.recs.size());
        cudaSetDevice(M[k].dev);
        e = cudaMemcpy(o.data(), M[k].out, o.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { gpu_note("fdtd multi probe copy-back", e); rc = 31; break; }
        for (int32_t it = 0; it < n_steps; ++it)
            for (size_t r = 0; r < s.recs.size(); ++r)
                out[(size_t)it * n_rec + s.recs[r]] = o[(size_t)it * s.recs.size() + r];
    }
    sess_free_run(X);
    return rc;
}
#else
struct MSession {
    std::vector<MSlab> S;
    std::vector<int32_t> devs;
    int64_t n_air = 0;
    bool have_f = false, have_r = false;
    float c4c = 0.f, c4nb = 0.f, c4nb2 = 0.f, lam2 = 0.f;
};
#endif

}  // namespace

extern "C" {

/* Session: a grid kept on its devices, created from the run4m arguments minus source, probes
 * and length (status: 0 or the run4m codes); then any number of aur_fdtd_session4m_run, then
 * aur_fdtd_session4m_destroy. One device: n_dev = 1, cuts = halo = {0, n_air} (run4 bit for bit). */
static void* sess_create_impl(int32_t n_dev, const int32_t* dev_ids,
                              const int64_t* cuts, const int64_t* halo,
                              const int32_t* nb, const int32_t* nb2,
                              const uint8_t* mask4, const float* c_now,
                              const float* c_old, const float* c_nb,
                              const float* c_drv, float c4c, float c4nb, float c4nb2,
                              float lam2, int64_t n_air,
                              int64_t n_faces, const int32_t* f_cell,
                              const float* f_e, const float* f_g,
                              const float* f_l, const float* f_Asum,
                              int64_t n_rfaces, const int32_t* r_cell,
                              const float* r_A, const float* r_c1,
                              const float* r_c2, const float* r_E,
                              const float* r_G,
                              const int64_t* pml_counts, const int32_t* pml_cells,
                              const float* pml_b, const float* pml_a,
                              const float* pml_apsi, const int32_t* pml_nbm,
                              const int32_t* pml_nbp, const int32_t* pml_locm,
                              int64_t n_lrows, const int32_t* l_rows, const int64_t* l_ptr,
                              const int32_t* l_nb, const float* l_k, int32_t* status) {
    int32_t dummy = 0;
    int32_t& st = status ? *status : dummy;
    st = 0;
    if (n_dev < 1 || !dev_ids || !cuts || !halo || !nb || !nb2 || !mask4 || !c_now
        || n_air <= 0) { st = 1; return nullptr; }
    if (n_lrows > 0 && (!l_rows || !l_ptr || !l_nb || !l_k)) { st = 43; return nullptr; }
    MSession* X = new MSession();
    X->devs.assign(dev_ids, dev_ids + n_dev);
    X->n_air = n_air;
    X->have_f = (n_faces > 0 && f_cell && c_drv);
    X->have_r = (n_rfaces > 0 && r_cell && c_drv);
    X->c4c = c4c; X->c4nb = c4nb; X->c4nb2 = c4nb2; X->lam2 = lam2;
    MArgs A{nb, nb2, mask4, c_now, c_old, c_nb, c_drv, n_air,
            X->have_f ? n_faces : 0, f_cell, f_e, f_g, f_l, f_Asum,
            X->have_r ? n_rfaces : 0, r_cell, r_A, r_c1, r_c2, r_E, r_G,
            pml_counts, pml_cells, pml_b, pml_a, pml_apsi, pml_nbm, pml_nbp, pml_locm,
            n_lrows > 0 ? n_lrows : 0, l_rows, l_ptr, l_nb, l_k};
    st = plan_bounds(n_dev, cuts, halo, n_air, X->S);
#if AUR_GPU
    if (st == 0) {
        cudaError_t e = cudaFree(0);
        if (e != cudaSuccess) { gpu_note("fdtd context init", e); st = 30; }
    }
    if (st == 0) st = sess_create_gpu(*X, A);
    if (st != 0) { sess_destroy_gpu(*X); delete X; return nullptr; }
#else
    if (st == 0)
        for (auto& sl : X->S)
            if ((st = fill_mslab(sl, A)) != 0) break;
    if (st != 0) { delete X; return nullptr; }
#endif
    return X;
}

AUR_API void* aur_fdtd_session4m_create(int32_t n_dev, const int32_t* dev_ids,
                                        const int64_t* cuts, const int64_t* halo,
                                        const int32_t* nb, const int32_t* nb2,
                                        const uint8_t* mask4, const float* c_now,
                                        const float* c_old, const float* c_nb,
                                        const float* c_drv, float c4c, float c4nb, float c4nb2,
                                        float lam2, int64_t n_air,
                                        int64_t n_faces, const int32_t* f_cell,
                                        const float* f_e, const float* f_g,
                                        const float* f_l, const float* f_Asum,
                                        int64_t n_rfaces, const int32_t* r_cell,
                                        const float* r_A, const float* r_c1,
                                        const float* r_c2, const float* r_E,
                                        const float* r_G,
                                        const int64_t* pml_counts, const int32_t* pml_cells,
                                        const float* pml_b, const float* pml_a,
                                        const float* pml_apsi, const int32_t* pml_nbm,
                                        const int32_t* pml_nbp, const int32_t* pml_locm,
                                        int64_t n_lrows, int32_t* status) {
    if (n_lrows > 0) {                  /* no link arrays here: aur_fdtd_session5m_create */
        if (status) *status = 43;
        return nullptr;
    }
    return sess_create_impl(n_dev, dev_ids, cuts, halo, nb, nb2, mask4, c_now, c_old, c_nb,
                            c_drv, c4c, c4nb, c4nb2, lam2, n_air, n_faces, f_cell, f_e, f_g,
                            f_l, f_Asum, n_rfaces, r_cell, r_A, r_c1, r_c2, r_E, r_G,
                            pml_counts, pml_cells, pml_b, pml_a, pml_apsi, pml_nbm, pml_nbp,
                            pml_locm, 0, nullptr, nullptr, nullptr, nullptr, status);
}

/* session4m_create with the tangential links of conformal_aniso (run5 CSR: l_rows, l_ptr
 * (n_lrows + 1), l_nb, l_k), applied right after the stencil as run5 (42: a link leaves its
 * slab's copy). Bit for bit against one domain. */
AUR_API void* aur_fdtd_session5m_create(int32_t n_dev, const int32_t* dev_ids,
                                        const int64_t* cuts, const int64_t* halo,
                                        const int32_t* nb, const int32_t* nb2,
                                        const uint8_t* mask4, const float* c_now,
                                        const float* c_old, const float* c_nb,
                                        const float* c_drv, float c4c, float c4nb, float c4nb2,
                                        float lam2, int64_t n_air,
                                        int64_t n_faces, const int32_t* f_cell,
                                        const float* f_e, const float* f_g,
                                        const float* f_l, const float* f_Asum,
                                        int64_t n_rfaces, const int32_t* r_cell,
                                        const float* r_A, const float* r_c1,
                                        const float* r_c2, const float* r_E,
                                        const float* r_G,
                                        const int64_t* pml_counts, const int32_t* pml_cells,
                                        const float* pml_b, const float* pml_a,
                                        const float* pml_apsi, const int32_t* pml_nbm,
                                        const int32_t* pml_nbp, const int32_t* pml_locm,
                                        int64_t n_lrows, const int32_t* l_rows,
                                        const int64_t* l_ptr, const int32_t* l_nb,
                                        const float* l_k, int32_t* status) {
    return sess_create_impl(n_dev, dev_ids, cuts, halo, nb, nb2, mask4, c_now, c_old, c_nb,
                            c_drv, c4c, c4nb, c4nb2, lam2, n_air, n_faces, f_cell, f_e, f_g,
                            f_l, f_Asum, n_rfaces, r_cell, r_A, r_c1, r_c2, r_E, r_G,
                            pml_counts, pml_cells, pml_b, pml_a, pml_apsi, pml_nbm, pml_nbp,
                            pml_locm, n_lrows, l_rows, l_ptr, l_nb, l_k, status);
}

/* One run on a session: the source taps, the probes (8 taps each, ghost = n_air), the pulse
 * and the length. 0, or the run4m codes (41: instability). */
AUR_API int32_t aur_fdtd_session4m_run(void* sess, const float* pulse, int32_t n_pulse,
                                       const int32_t* src_idx, const float* src_w,
                                       int32_t n_src, const int32_t* rec_idx,
                                       const float* rec_w, int32_t n_rec, int32_t n_steps,
                                       float* out) {
    if (!sess || n_steps <= 0 || !out) return 1;
    MSession& X = *static_cast<MSession*>(sess);
    int32_t rc = assign_points(X.S, X.n_air, src_idx, src_w, n_src, rec_idx, rec_w, n_rec);
    if (rc) return rc;
#if AUR_GPU
    return sess_run_gpu(X, pulse, n_pulse, n_rec, n_steps, out);
#else
    return run4m_host(X.S, X.have_f, X.have_r, X.c4c, X.c4nb, X.c4nb2, X.lam2, pulse, n_pulse,
                      n_rec, n_steps, out);
#endif
}

AUR_API void aur_fdtd_session4m_destroy(void* sess) {
    if (!sess) return;
    MSession* X = static_cast<MSession*>(sess);
#if AUR_GPU
    sess_destroy_gpu(*X);
#endif
    delete X;
}

/* Slab-by-slab session, for scenes above 2^31 air cells: each slab comes from its own sub grid
 * (whole x planes, cells from 0, ghost = its cell count), never a whole-scene grid. cuts
 * (n_dev + 1) and halo (2 n_dev) use the VIRTUAL numbering (owned cells end to end, n_virtual
 * in all). aur_fdtd_slabs_add(k, shift = virtual index of the sub grid's cell 0, arrays) fills
 * and uploads slab k; aur_fdtd_session6m_run takes int64 virtual indices;
 * aur_fdtd_session4m_destroy ends it. */
AUR_API void* aur_fdtd_slabs_create(int32_t n_dev, const int32_t* dev_ids, const int64_t* cuts,
                                    const int64_t* halo, int64_t n_virtual, int32_t have_f,
                                    int32_t have_r, float c4c, float c4nb, float c4nb2,
                                    float lam2, int32_t* status) {
    int32_t dummy = 0;
    int32_t& st = status ? *status : dummy;
    st = 0;
    if (n_dev < 1 || !dev_ids || !cuts || !halo || n_virtual <= 0) { st = 1; return nullptr; }
    MSession* X = new MSession();
    X->devs.assign(dev_ids, dev_ids + n_dev);
    X->n_air = n_virtual;
    X->have_f = have_f != 0;
    X->have_r = have_r != 0;
    X->c4c = c4c; X->c4nb = c4nb; X->c4nb2 = c4nb2; X->lam2 = lam2;
    st = plan_bounds(n_dev, cuts, halo, n_virtual, X->S);
#if AUR_GPU
    if (st == 0) {
        cudaError_t e = cudaFree(0);
        if (e != cudaSuccess) { gpu_note("fdtd context init", e); st = 30; }
    }
    if (st == 0) {
        X->M.assign(X->S.size(), MDev());
        for (size_t k = 0; k < X->S.size(); ++k) X->M[k].dev = X->devs[k];
        sess_peers(*X);
    }
#endif
    if (st != 0) { delete X; return nullptr; }
    return X;
}

AUR_API int32_t aur_fdtd_slabs_add(void* sess, int32_t k, int64_t shift, int64_t n_sub,
                                   const int32_t* nb, const int32_t* nb2, const uint8_t* mask4,
                                   const float* c_now, const float* c_old, const float* c_nb,
                                   const float* c_drv, int64_t n_faces, const int32_t* f_cell,
                                   const float* f_e, const float* f_g, const float* f_l,
                                   const float* f_Asum, int64_t n_rfaces, const int32_t* r_cell,
                                   const float* r_A, const float* r_c1, const float* r_c2,
                                   const float* r_E, const float* r_G, const int64_t* pml_counts,
                                   const int32_t* pml_cells, const float* pml_b,
                                   const float* pml_a, const float* pml_apsi,
                                   const int32_t* pml_nbm, const int32_t* pml_nbp,
                                   const int32_t* pml_locm, int64_t n_lrows,
                                   const int32_t* l_rows, const int64_t* l_ptr,
                                   const int32_t* l_nb, const float* l_k) {
    if (!sess) return 1;
    MSession& X = *static_cast<MSession*>(sess);
    if (k < 0 || k >= (int32_t)X.S.size() || !nb || !nb2 || !mask4 || !c_now || n_sub <= 0)
        return 1;
    if (n_lrows > 0 && (!l_rows || !l_ptr || !l_nb || !l_k)) return 43;
    MArgs A{nb, nb2, mask4, c_now, c_old, c_nb, c_drv, X.n_air,
            X.have_f ? n_faces : 0, f_cell, f_e, f_g, f_l, f_Asum,
            X.have_r ? n_rfaces : 0, r_cell, r_A, r_c1, r_c2, r_E, r_G,
            pml_counts, pml_cells, pml_b, pml_a, pml_apsi, pml_nbm, pml_nbp, pml_locm,
            n_lrows > 0 ? n_lrows : 0, l_rows, l_ptr, l_nb, l_k, shift, n_sub};
    MSlab& s = X.S[(size_t)k];
    if (s.hl - shift < 0 || s.hr - shift > n_sub) return 42;
    if (int32_t rc = fill_mslab(s, A)) return rc;
#if AUR_GPU
    MDev& d = X.M[(size_t)k];
    cudaError_t e = cudaSetDevice(d.dev);
    if (e != cudaSuccess) { gpu_note("fdtd slabs set device", e); return 31; }
    return upload_mslab(s, d, X.have_f || X.have_r);
#else
    return 0;
#endif
}

AUR_API int32_t aur_fdtd_session6m_run(void* sess, const float* pulse, int32_t n_pulse,
                                       const int64_t* src_idx, const float* src_w,
                                       int32_t n_src, const int64_t* rec_idx,
                                       const float* rec_w, int32_t n_rec, int32_t n_steps,
                                       float* out) {
    if (!sess || n_steps <= 0 || !out) return 1;
    MSession& X = *static_cast<MSession*>(sess);
    int32_t rc = assign_points(X.S, X.n_air, src_idx, src_w, n_src, rec_idx, rec_w, n_rec);
    if (rc) return rc;
#if AUR_GPU
    return sess_run_gpu(X, pulse, n_pulse, n_rec, n_steps, out);
#else
    return run4m_host(X.S, X.have_f, X.have_r, X.c4c, X.c4nb, X.c4nb2, X.lam2, pulse, n_pulse,
                      n_rec, n_steps, out);
#endif
}

/* run4 over n_dev devices, through a one-run session. dev_ids (n_dev), cuts (n_dev + 1 air-cell
 * indices at x-plane starts: slab k owns [cuts[k], cuts[k+1])), halo (2 n_dev: the [hl, hr) copy
 * of each slab, two x planes each side). 42: a cut through the x PML or a slab too thin;
 * 43: tangential links, use run5m. */
AUR_API int32_t aur_fdtd_run4m(int32_t n_dev, const int32_t* dev_ids, const int64_t* cuts,
                               const int64_t* halo,
                               const int32_t* nb, const int32_t* nb2, const uint8_t* mask4,
                               const float* c_now, const float* c_old,
                               const float* c_nb, const float* c_drv,
                               float c4c, float c4nb, float c4nb2,
                               float lam2, int64_t n_air,
                               int64_t n_faces, const int32_t* f_cell,
                               const float* f_e, const float* f_g,
                               const float* f_l, const float* f_Asum,
                               int64_t n_rfaces, const int32_t* r_cell,
                               const float* r_A, const float* r_c1,
                               const float* r_c2, const float* r_E,
                               const float* r_G,
                               const int64_t* pml_counts, const int32_t* pml_cells,
                               const float* pml_b, const float* pml_a, const float* pml_apsi,
                               const int32_t* pml_nbm, const int32_t* pml_nbp,
                               const int32_t* pml_locm, int64_t n_lrows,
                               const float* pulse, int32_t n_pulse,
                               const int32_t* src_idx, const float* src_w, int32_t n_src,
                               const int32_t* rec_idx, const float* rec_w, int32_t n_rec,
                               int32_t n_steps, int32_t n_threads, float* out) {
    (void)n_threads;
    if (n_steps <= 0 || !out) return 1;
    int32_t st = 0;
    void* X = aur_fdtd_session4m_create(n_dev, dev_ids, cuts, halo, nb, nb2, mask4, c_now,
                                        c_old, c_nb, c_drv, c4c, c4nb, c4nb2, lam2, n_air,
                                        n_faces, f_cell, f_e, f_g, f_l, f_Asum, n_rfaces, r_cell,
                                        r_A, r_c1, r_c2, r_E, r_G, pml_counts, pml_cells, pml_b,
                                        pml_a, pml_apsi, pml_nbm, pml_nbp, pml_locm, n_lrows, &st);
    if (!X) return st ? st : 31;
    const int32_t rc = aur_fdtd_session4m_run(X, pulse, n_pulse, src_idx, src_w, n_src,
                                              rec_idx, rec_w, n_rec, n_steps, out);
    aur_fdtd_session4m_destroy(X);
    return rc;
}

/* run4m with the tangential links (conformal_aniso), through aur_fdtd_session5m_create. */
AUR_API int32_t aur_fdtd_run5m(int32_t n_dev, const int32_t* dev_ids, const int64_t* cuts,
                               const int64_t* halo,
                               const int32_t* nb, const int32_t* nb2, const uint8_t* mask4,
                               const float* c_now, const float* c_old,
                               const float* c_nb, const float* c_drv,
                               float c4c, float c4nb, float c4nb2,
                               float lam2, int64_t n_air,
                               int64_t n_faces, const int32_t* f_cell,
                               const float* f_e, const float* f_g,
                               const float* f_l, const float* f_Asum,
                               int64_t n_rfaces, const int32_t* r_cell,
                               const float* r_A, const float* r_c1,
                               const float* r_c2, const float* r_E,
                               const float* r_G,
                               const int64_t* pml_counts, const int32_t* pml_cells,
                               const float* pml_b, const float* pml_a, const float* pml_apsi,
                               const int32_t* pml_nbm, const int32_t* pml_nbp,
                               const int32_t* pml_locm,
                               int64_t n_lrows, const int32_t* l_rows,
                               const int64_t* l_ptr, const int32_t* l_nb, const float* l_k,
                               const float* pulse, int32_t n_pulse,
                               const int32_t* src_idx, const float* src_w, int32_t n_src,
                               const int32_t* rec_idx, const float* rec_w, int32_t n_rec,
                               int32_t n_steps, int32_t n_threads, float* out) {
    (void)n_threads;
    if (n_steps <= 0 || !out) return 1;
    int32_t st = 0;
    void* X = aur_fdtd_session5m_create(n_dev, dev_ids, cuts, halo, nb, nb2, mask4, c_now,
                                        c_old, c_nb, c_drv, c4c, c4nb, c4nb2, lam2, n_air,
                                        n_faces, f_cell, f_e, f_g, f_l, f_Asum, n_rfaces, r_cell,
                                        r_A, r_c1, r_c2, r_E, r_G, pml_counts, pml_cells, pml_b,
                                        pml_a, pml_apsi, pml_nbm, pml_nbp, pml_locm,
                                        n_lrows, l_rows, l_ptr, l_nb, l_k, &st);
    if (!X) return st ? st : 31;
    const int32_t rc = aur_fdtd_session4m_run(X, pulse, n_pulse, src_idx, src_w, n_src,
                                              rec_idx, rec_w, n_rec, n_steps, out);
    aur_fdtd_session4m_destroy(X);
    return rc;
}

/* Dense run of an EMPTY box (second pass of open domains), no neighbour tables: p0, p1, p2 on
 * the box padded by two zero cells (12 bytes a cell), int64 indices. Order-4 cells (2+ cells from
 * the faces, outside the CPML and its collar) take c4*, the others the order-2 coefficients of
 * their count m of box faces (coef[3 m .. 3 m + 2] = c_now, c_nb, c_old). CPML entries as in
 * FdtdGrid, operations in the order of aur_fdtd_run4: the same floats bit for bit. */
struct DenseBox {
    int64_t nx, ny, nz, npml, PY, PZ;
    DEVMETH int64_t at(int64_t i, int64_t j, int64_t k) const {
        return ((i + 2) * PY + (j + 2)) * PZ + (k + 2);
    }
    DEVMETH bool core(int64_t ia, int64_t n) const {
        return npml <= 0 || (ia >= npml + 1 && ia <= n - npml - 2);
    }
};

DEVFN void dense_cell(const DenseBox& B, int64_t t, float c4c, float c4nb, float c4nb2,
                      const float* coef, const float* p0, const float* p1, float* p2) {
    const int64_t k = t % B.nz, r = t / B.nz, j = r % B.ny, i = r / B.ny;
    const int64_t q = B.at(i, j, k), sx = B.PY * B.PZ, sy = B.PZ;
    const float sq = p1[q - sx] + p1[q + sx] + p1[q - sy] + p1[q + sy] + p1[q - 1] + p1[q + 1];
    int64_t d = i < B.nx - 1 - i ? i : B.nx - 1 - i;
    const int64_t dj = j < B.ny - 1 - j ? j : B.ny - 1 - j;
    const int64_t dk = k < B.nz - 1 - k ? k : B.nz - 1 - k;
    if (dj < d) d = dj;
    if (dk < d) d = dk;
    if (d >= 2 && B.core(i, B.nx) && B.core(j, B.ny) && B.core(k, B.nz)) {
        const float s2 = p1[q - 2 * sx] + p1[q + 2 * sx] + p1[q - 2 * sy] + p1[q + 2 * sy]
                       + p1[q - 2] + p1[q + 2];
        p2[q] = c4c * p1[q] + c4nb * sq + c4nb2 * s2 - p0[q];
    } else {
        const int m = (i == 0) + (i == B.nx - 1) + (j == 0) + (j == B.ny - 1) + (k == 0)
                    + (k == B.nz - 1);
        const float* c = coef + 3 * (m < 3 ? m : 3);
        p2[q] = c[0] * p1[q] + c[1] * sq + c[2] * p0[q];
    }
}

DEVFN void dense_psi(int64_t e, const int64_t* pc, const int64_t* pp, const float* pb,
                     const float* pap, const float* p1, float* psi) {
    const float dh = p1[pp[e]] - p1[pc[e]];
    psi[e] = pb[e] * psi[e] + pap[e] * dh;
}

DEVFN void dense_apply(int64_t o, int64_t k, const int64_t* pc, const int64_t* pm,
                       const int64_t* pp, const int64_t* pl, const float* pb, const float* pa,
                       const float* psi, float* phiP, float lam2, const float* p1, float* p2) {
    const int64_t g = o + k;
    const int64_t lm = pl[g];
    const float pmv = (lm >= 0) ? psi[o + lm] : 0.f;
    const float dpsi = psi[g] - pmv;
    const int64_t cc = pc[g];
    const float d2 = p1[pp[g]] - 2.f * p1[cc] + p1[pm[g]];
    const float ph = pb[g] * phiP[g] + pa[g] * (d2 + dpsi);
    phiP[g] = ph;
    p2[cc] += lam2 * (dpsi + ph);
}

#if AUR_GPU
GLOBALFN void aur_dense_step(DenseBox B, int64_t n, float c4c, float c4nb, float c4nb2,
                             const float* coef, const float* p0, const float* p1, float* p2) {
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (int64_t)gridDim.x * blockDim.x)
        dense_cell(B, t, c4c, c4nb, c4nb2, coef, p0, p1, p2);
}
GLOBALFN void aur_dense_step_range(DenseBox B, int64_t t0, int64_t n, float c4c, float c4nb,
                                   float c4nb2, const float* coef, const float* p0,
                                   const float* p1, float* p2) {
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (int64_t)gridDim.x * blockDim.x)
        dense_cell(B, t0 + t, c4c, c4nb, c4nb2, coef, p0, p1, p2);
}
GLOBALFN void aur_dense_psi(int64_t n, const int64_t* pc, const int64_t* pp, const float* pb,
                            const float* pap, const float* p1, float* psi) {
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (int64_t)gridDim.x * blockDim.x)
        dense_psi(e, pc, pp, pb, pap, p1, psi);
}
GLOBALFN void aur_dense_apply(int64_t o, int64_t cnt, const int64_t* pc, const int64_t* pm,
                              const int64_t* pp, const int64_t* pl, const float* pb,
                              const float* pa, const float* psi, float* phiP, float lam2,
                              const float* p1, float* p2) {
    for (int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; k < cnt;
         k += (int64_t)gridDim.x * blockDim.x)
        dense_apply(o, k, pc, pm, pp, pl, pb, pa, psi, phiP, lam2, p1, p2);
}
GLOBALFN void aur_dense_io(float* p2, const int64_t* si, const float* sw, int32_t n_src,
                           float amp, int32_t inject, const int64_t* ri, const float* rw,
                           int32_t n_rec, float* out_step) {
    /* one role per launch (inject, or read): over as many blocks as needed */
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (inject && t < n_src) p2[si[t]] += amp * sw[t];
    if (t < n_rec) {
        float acc = 0.f;
        for (int k = 0; k < 8; ++k) acc += p2[ri[8 * t + k]] * rw[8 * t + k];
        out_step[t] = acc;
    }
}
#endif

AUR_API int32_t aur_fdtd_dense_empty(int64_t nx, int64_t ny, int64_t nz, int64_t npml,
                                     float c4c, float c4nb, float c4nb2, const float* coef,
                                     float lam2, const int64_t* pml_counts,
                                     const int64_t* pml_cells, const int64_t* pml_nbm,
                                     const int64_t* pml_nbp, const int64_t* pml_locm,
                                     const float* pml_b, const float* pml_a,
                                     const float* pml_apsi, const float* pulse, int32_t n_pulse,
                                     const int64_t* src_idx, const float* src_w, int32_t n_src,
                                     const int64_t* rec_idx, const float* rec_w, int32_t n_rec,
                                     int32_t n_steps, float* out) {
    if (nx < 3 || ny < 3 || nz < 3 || n_steps <= 0 || !out || !coef || n_src > 1024
        || n_rec > 1024) return 1;
    DenseBox B{nx, ny, nz, npml, ny + 4, nz + 4};
    const int64_t n = nx * ny * nz, np_ = (nx + 4) * B.PY * B.PZ;
    int64_t off[4] = {0, 0, 0, 0};
    if (pml_counts)
        for (int a = 0; a < 3; ++a) off[a + 1] = off[a] + pml_counts[a];
    const int64_t n_pml = off[3];
#if AUR_GPU
    cudaError_t e = cudaFree(0);
    if (e != cudaSuccess) { gpu_note("dense context init", e); return 30; }
    std::vector<void*> owned;
    float *P0 = nullptr, *P1 = nullptr, *P2 = nullptr, *dcoef = nullptr, *psi = nullptr,
          *phiP = nullptr, *pb = nullptr, *pa = nullptr, *pap = nullptr, *sw = nullptr,
          *rw = nullptr, *dout = nullptr;
    int64_t *pc = nullptr, *pm = nullptr, *pp = nullptr, *pl = nullptr, *si = nullptr,
            *ri = nullptr;
    auto zal = [&](void** d, size_t b) -> bool {
        if (!b) { *d = nullptr; return true; }
        e = cudaMalloc(d, b);
        if (e != cudaSuccess) { gpu_note("dense cudaMalloc", e); return false; }
        owned.push_back(*d);
        e = cudaMemset(*d, 0, b);
        return e == cudaSuccess;
    };
    auto up = [&](void** d, const void* h, size_t b) -> bool {
        if (!zal(d, b)) return false;
        if (!b) return true;
        e = cudaMemcpy(*d, h, b, cudaMemcpyHostToDevice);
        return e == cudaSuccess;
    };
    const int32_t rc_fail = 31;
    bool ok = zal((void**)&P0, np_ * sizeof(float)) && zal((void**)&P1, np_ * sizeof(float))
        && zal((void**)&P2, np_ * sizeof(float)) && up((void**)&dcoef, coef, 12 * sizeof(float))
        && up((void**)&pc, pml_cells, n_pml * sizeof(int64_t))
        && up((void**)&pm, pml_nbm, n_pml * sizeof(int64_t))
        && up((void**)&pp, pml_nbp, n_pml * sizeof(int64_t))
        && up((void**)&pl, pml_locm, n_pml * sizeof(int64_t))
        && up((void**)&pb, pml_b, n_pml * sizeof(float))
        && up((void**)&pa, pml_a, n_pml * sizeof(float))
        && up((void**)&pap, pml_apsi, n_pml * sizeof(float))
        && zal((void**)&psi, n_pml * sizeof(float)) && zal((void**)&phiP, n_pml * sizeof(float))
        && up((void**)&si, src_idx, (size_t)n_src * sizeof(int64_t))
        && up((void**)&sw, src_w, (size_t)n_src * sizeof(float))
        && up((void**)&ri, rec_idx, 8 * (size_t)n_rec * sizeof(int64_t))
        && up((void**)&rw, rec_w, 8 * (size_t)n_rec * sizeof(float))
        && zal((void**)&dout, (size_t)n_steps * (n_rec > 0 ? n_rec : 1) * sizeof(float));
    if (!ok) { for (void* q : owned) cudaFree(q); return rc_fail; }
    const int BS = 256;
    auto blocks = [&](int64_t m) { int64_t b = (m + BS - 1) / BS; return (int)(b < 1048576 ? (b > 0 ? b : 1) : 1048576); };
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        aur_dense_step<<<blocks(n), BS>>>(B, n, c4c, c4nb, c4nb2, dcoef, P0, P1, P2);
        if (n_pml > 0) {
            aur_dense_psi<<<blocks(n_pml), BS>>>(n_pml, pc, pp, pb, pap, P1, psi);
            for (int a = 0; a < 3; ++a) {
                const int64_t cnt = off[a + 1] - off[a];
                if (cnt > 0)
                    aur_dense_apply<<<blocks(cnt), BS>>>(off[a], cnt, pc, pm, pp, pl, pb, pa,
                                                        psi, phiP, lam2, P1, P2);
            }
        }
        /* the injection, then the reading (two launches on one stream: in order) */
        if (n_src > 0 && it < n_pulse)
            aur_dense_io<<<io_blocks(n_src), 256>>>(P2, si, sw, n_src, pulse[it], 1, nullptr,
                                                    nullptr, 0, nullptr);
        if (n_rec > 0)
            aur_dense_io<<<io_blocks(n_rec), 256>>>(P2, nullptr, nullptr, 0, 0.f, 0, ri, rw,
                                                    n_rec, dout + (size_t)it * n_rec);
        float* t = P0; P0 = P1; P1 = P2; P2 = t;
    }
    e = cudaDeviceSynchronize();
    if (e == cudaSuccess && n_rec > 0)
        e = cudaMemcpy(out, dout, (size_t)n_steps * n_rec * sizeof(float),
                       cudaMemcpyDeviceToHost);
    for (void* q : owned) cudaFree(q);
    if (e != cudaSuccess) { gpu_note("dense run", e); return 200 + (int)e; }
    return 0;
#else
    std::vector<float> P0((size_t)np_, 0.f), P1((size_t)np_, 0.f), P2((size_t)np_, 0.f);
    std::vector<float> psi((size_t)n_pml, 0.f), phiP((size_t)n_pml, 0.f);
    float* a0 = P0.data(); float* a1 = P1.data(); float* a2 = P2.data();
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        for (int64_t t = 0; t < n; ++t) dense_cell(B, t, c4c, c4nb, c4nb2, coef, a0, a1, a2);
        if (n_pml > 0) {
            for (int64_t q = 0; q < n_pml; ++q)
                dense_psi(q, pml_cells, pml_nbp, pml_b, pml_apsi, a1, psi.data());
            for (int a = 0; a < 3; ++a)
                for (int64_t k = 0; k < off[a + 1] - off[a]; ++k)
                    dense_apply(off[a], k, pml_cells, pml_nbm, pml_nbp, pml_locm, pml_b, pml_a,
                                psi.data(), phiP.data(), lam2, a1, a2);
        }
        if (it < n_pulse)
            for (int32_t q = 0; q < n_src; ++q) a2[src_idx[q]] += pulse[it] * src_w[q];
        for (int32_t r = 0; r < n_rec; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 8; ++k) acc += a2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
            out[(size_t)it * n_rec + r] = acc;
        }
        float* t = a0; a0 = a1; a1 = a2; a2 = t;
    }
    return 0;
#endif
}

/* Dense empty box split over n_dev devices in slabs of whole x planes: slab k owns the real
 * planes [xcuts[k], xcuts[k+1]) plus a copy of two planes each side, refreshed every step. CPML
 * entries, source taps and receivers (by first tap) go to the slab owning their cell. The same
 * floats as aur_fdtd_dense_empty. 42: a cut through the x CPML, or a receiver across a slab. */
AUR_API int32_t aur_fdtd_dense_empty_m(int32_t n_dev, const int32_t* dev_ids,
                                       const int64_t* xcuts, int64_t nx, int64_t ny, int64_t nz,
                                       int64_t npml, float c4c, float c4nb, float c4nb2,
                                       const float* coef, float lam2, const int64_t* pml_counts,
                                       const int64_t* pml_cells, const int64_t* pml_nbm,
                                       const int64_t* pml_nbp, const int64_t* pml_locm,
                                       const float* pml_b, const float* pml_a,
                                       const float* pml_apsi, const float* pulse,
                                       int32_t n_pulse, const int64_t* src_idx,
                                       const float* src_w, int32_t n_src,
                                       const int64_t* rec_idx, const float* rec_w, int32_t n_rec,
                                       int32_t n_steps, float* out) {
    if (n_dev < 1 || !dev_ids || !xcuts || nx < 3 || ny < 3 || nz < 3 || n_steps <= 0 || !out
        || !coef || n_src > 1024 || n_rec > 1024) return 1;
    DenseBox B{nx, ny, nz, npml, ny + 4, nz + 4};
    const int64_t sx = B.PY * B.PZ;
    int64_t off[4] = {0, 0, 0, 0};
    if (pml_counts)
        for (int a = 0; a < 3; ++a) off[a + 1] = off[a] + pml_counts[a];
    for (int32_t k = 0; k < n_dev; ++k)
        if (!(xcuts[k] < xcuts[k + 1]) || xcuts[k + 1] - xcuts[k] < 2) return 42;
    if (xcuts[0] != 0 || xcuts[n_dev] != nx) return 42;
    struct DS {
        int64_t x0, x1, base;                       /* owned real planes, padded flat of local 0 */
        int64_t eoff[4];                            /* its CPML entries, per axis */
        std::vector<int64_t> pc, pm, pp, pl, si, ri;
        std::vector<float> pb, pa, pap, sw, rw;
        std::vector<int32_t> recs;
    };
    std::vector<DS> S((size_t)n_dev);
    auto plane = [&](int64_t q) { return q / sx - 2; };   /* real plane of a padded flat index */
    for (int32_t k = 0; k < n_dev; ++k) {
        DS& d = S[(size_t)k];
        d.x0 = xcuts[k]; d.x1 = xcuts[k + 1]; d.base = d.x0 * sx;
        const int64_t q_lo = (d.x0 + 2) * sx, q_hi = (d.x1 + 2) * sx;
        for (int a = 0; a < 3; ++a) {
            d.eoff[a] = (int64_t)d.pc.size();
            const int64_t* cb = pml_cells + off[a];
            const int64_t cnt = off[a + 1] - off[a];
            const int64_t lo = std::lower_bound(cb, cb + cnt, q_lo) - cb;
            const int64_t hi = std::lower_bound(cb, cb + cnt, q_hi) - cb;
            for (int64_t j = lo; j < hi; ++j) {
                const int64_t g = off[a] + j;
                const int64_t lm = pml_locm[g];
                if (lm >= 0 && (lm < lo || lm >= hi)) return 42;
                d.pc.push_back(pml_cells[g]); d.pm.push_back(pml_nbm[g]);
                d.pp.push_back(pml_nbp[g]); d.pl.push_back(lm >= 0 ? lm - lo : -1);
                d.pb.push_back(pml_b[g]); d.pa.push_back(pml_a[g]); d.pap.push_back(pml_apsi[g]);
            }
        }
        d.eoff[3] = (int64_t)d.pc.size();
        for (int32_t q = 0; q < n_src; ++q) {
            const int64_t x = plane(src_idx[q]);
            if (x >= d.x0 && x < d.x1) { d.si.push_back(src_idx[q]); d.sw.push_back(src_w[q]); }
        }
    }
    for (int32_t r = 0; r < n_rec; ++r) {
        const int64_t x = plane(rec_idx[8 * r]);
        int32_t k = 0;
        while (k + 1 < n_dev && x >= xcuts[k + 1]) ++k;
        DS& d = S[(size_t)k];
        for (int q = 0; q < 8; ++q) {
            const int64_t xq = plane(rec_idx[8 * r + q]);
            if (rec_w[8 * r + q] != 0.f && (xq < d.x0 - 2 || xq >= d.x1 + 2)) return 42;
            d.ri.push_back(rec_w[8 * r + q] != 0.f ? rec_idx[8 * r + q] : d.base);
            d.rw.push_back(rec_w[8 * r + q]);
        }
        d.recs.push_back(r);
    }
#if AUR_GPU
    cudaError_t e = cudaFree(0);
    if (e != cudaSuccess) { gpu_note("dense multi context init", e); return 30; }
    struct DD {
        int dev = 0; std::vector<void*> owned;
        float *P0 = nullptr, *P1 = nullptr, *P2 = nullptr, *coef = nullptr, *psi = nullptr,
              *phiP = nullptr, *pb = nullptr, *pa = nullptr, *pap = nullptr, *sw = nullptr,
              *rw = nullptr, *out = nullptr;
        int64_t *pc = nullptr, *pm = nullptr, *pp = nullptr, *pl = nullptr, *si = nullptr,
                *ri = nullptr;
        cudaStream_t st = nullptr; cudaEvent_t upd = nullptr, cpy = nullptr;
    };
    std::vector<DD> M((size_t)n_dev);
    bool ok = true;
    for (int32_t k = 0; k < n_dev && ok; ++k) {
        DS& d = S[(size_t)k]; DD& m = M[(size_t)k];
        m.dev = dev_ids[k];
        if (cudaSetDevice(m.dev) != cudaSuccess) { ok = false; break; }
        for (int j : {k - 1, k + 1}) {
            if (j < 0 || j >= n_dev || dev_ids[j] == m.dev) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, m.dev, dev_ids[j]);
            if (can && cudaDeviceEnablePeerAccess(dev_ids[j], 0) != cudaSuccess) cudaGetLastError();
        }
        const size_t nl = (size_t)(d.x1 - d.x0 + 4) * (size_t)sx;
        auto zal = [&](void** p, size_t b) -> bool {
            if (!b) { *p = nullptr; return true; }
            if (cudaMalloc(p, b) != cudaSuccess) { gpu_note("dense multi cudaMalloc", cudaErrorMemoryAllocation); return false; }
            m.owned.push_back(*p);
            return cudaMemset(*p, 0, b) == cudaSuccess;
        };
        auto up = [&](void** p, const void* h, size_t b) -> bool {
            if (!zal(p, b)) return false;
            return !b || cudaMemcpy(*p, h, b, cudaMemcpyHostToDevice) == cudaSuccess;
        };
        const size_t ne = d.pc.size();
        ok = zal((void**)&m.P0, nl * 4) && zal((void**)&m.P1, nl * 4) && zal((void**)&m.P2, nl * 4)
            && up((void**)&m.coef, coef, 12 * 4)
            && up((void**)&m.pc, d.pc.data(), ne * 8) && up((void**)&m.pm, d.pm.data(), ne * 8)
            && up((void**)&m.pp, d.pp.data(), ne * 8) && up((void**)&m.pl, d.pl.data(), ne * 8)
            && up((void**)&m.pb, d.pb.data(), ne * 4) && up((void**)&m.pa, d.pa.data(), ne * 4)
            && up((void**)&m.pap, d.pap.data(), ne * 4) && zal((void**)&m.psi, ne * 4)
            && zal((void**)&m.phiP, ne * 4)
            && up((void**)&m.si, d.si.data(), d.si.size() * 8) && up((void**)&m.sw, d.sw.data(), d.sw.size() * 4)
            && up((void**)&m.ri, d.ri.data(), d.ri.size() * 8) && up((void**)&m.rw, d.rw.data(), d.rw.size() * 4)
            && zal((void**)&m.out, (size_t)n_steps * (d.recs.empty() ? 1 : d.recs.size()) * 4);
        if (ok) {
            /* global padded indexing: the local array is the padded planes [x0, x1 + 4) */
            m.P0 -= d.base; m.P1 -= d.base; m.P2 -= d.base;
            cudaStreamCreate(&m.st); cudaEventCreate(&m.upd); cudaEventCreate(&m.cpy);
        }
    }
    const int BS = 256;
    auto blocks = [&](int64_t q) { int64_t b = (q + BS - 1) / BS; return (int)(b < 1048576 ? (b > 0 ? b : 1) : 1048576); };
    for (int32_t it = 0; it < n_steps && ok; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        for (int32_t k = 0; k < n_dev; ++k) {
            DS& d = S[(size_t)k]; DD& m = M[(size_t)k];
            cudaSetDevice(m.dev);
            if (it > 0) cudaStreamWaitEvent(m.st, M[(size_t)k].cpy, 0);
            if (k > 0 && it > 0) cudaStreamWaitEvent(m.st, M[(size_t)k - 1].cpy, 0);
            if (k + 1 < n_dev && it > 0) cudaStreamWaitEvent(m.st, M[(size_t)k + 1].cpy, 0);
            const int64_t t0 = d.x0 * ny * nz, n = (d.x1 - d.x0) * ny * nz;
            aur_dense_step_range<<<blocks(n), BS, 0, m.st>>>(B, t0, n, c4c, c4nb, c4nb2, m.coef,
                                                            m.P0, m.P1, m.P2);
            const int64_t ne = d.eoff[3];
            if (ne > 0) {
                aur_dense_psi<<<blocks(ne), BS, 0, m.st>>>(ne, m.pc, m.pp, m.pb, m.pap, m.P1, m.psi);
                for (int a = 0; a < 3; ++a) {
                    const int64_t cnt = d.eoff[a + 1] - d.eoff[a];
                    if (cnt > 0)
                        aur_dense_apply<<<blocks(cnt), BS, 0, m.st>>>(d.eoff[a], cnt, m.pc, m.pm,
                                                                    m.pp, m.pl, m.pb, m.pa, m.psi,
                                                                    m.phiP, lam2, m.P1, m.P2);
                }
            }
            if (!d.si.empty() && it < n_pulse)
                aur_dense_io<<<io_blocks((int64_t)d.si.size()), 256, 0, m.st>>>(m.P2, m.si, m.sw, (int32_t)d.si.size(),
                                                              pulse[it], 1, nullptr, nullptr, 0,
                                                              nullptr);
            cudaEventRecord(m.upd, m.st);
        }
        for (int32_t k = 0; k < n_dev; ++k) {
            DS& d = S[(size_t)k]; DD& m = M[(size_t)k];
            cudaSetDevice(m.dev);
            /* the neighbours' edge planes into this slab's copies */
            if (k > 0) {
                cudaStreamWaitEvent(m.st, M[(size_t)k - 1].upd, 0);
                cudaMemcpyPeerAsync(m.P2 + (d.x0 - 2 + 2) * sx, m.dev,
                                    M[(size_t)k - 1].P2 + (d.x0 - 2 + 2) * sx, M[(size_t)k - 1].dev,
                                    (size_t)(2 * sx) * 4, m.st);
            }
            if (k + 1 < n_dev) {
                cudaStreamWaitEvent(m.st, M[(size_t)k + 1].upd, 0);
                cudaMemcpyPeerAsync(m.P2 + (d.x1 + 2) * sx, m.dev,
                                    M[(size_t)k + 1].P2 + (d.x1 + 2) * sx, M[(size_t)k + 1].dev,
                                    (size_t)(2 * sx) * 4, m.st);
            }
            if (!d.recs.empty())
                aur_dense_io<<<io_blocks((int64_t)d.recs.size()), 256, 0, m.st>>>(m.P2, nullptr, nullptr, 0, 0.f, 0,
                                                                m.ri, m.rw, (int32_t)d.recs.size(),
                                                                m.out + (size_t)it * d.recs.size());
            cudaEventRecord(m.cpy, m.st);
        }
        for (int32_t k = 0; k < n_dev; ++k) {
            DD& m = M[(size_t)k];
            float* t = m.P0; m.P0 = m.P1; m.P1 = m.P2; m.P2 = t;
        }
    }
    std::vector<float> buf;
    for (int32_t k = 0; k < n_dev && ok; ++k) {
        DS& d = S[(size_t)k]; DD& m = M[(size_t)k];
        cudaSetDevice(m.dev);
        if (cudaStreamSynchronize(m.st) != cudaSuccess) { ok = false; break; }
        if (d.recs.empty()) continue;
        buf.resize((size_t)n_steps * d.recs.size());
        if (cudaMemcpy(buf.data(), m.out, buf.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            ok = false; break;
        }
        for (int32_t it = 0; it < n_steps; ++it)
            for (size_t r = 0; r < d.recs.size(); ++r)
                out[(size_t)it * n_rec + d.recs[r]] = buf[(size_t)it * d.recs.size() + r];
    }
    for (DD& m : M) {
        cudaSetDevice(m.dev);
        if (m.st) { cudaStreamDestroy(m.st); cudaEventDestroy(m.upd); cudaEventDestroy(m.cpy); }
        for (void* p : m.owned) cudaFree(p);
    }
    return ok ? 0 : 31;
#else
    /* emulation: the slabs one after the other, the same operations and copies */
    std::vector<std::vector<float>> A0((size_t)n_dev), A1((size_t)n_dev), A2((size_t)n_dev),
        PSI((size_t)n_dev), PHI((size_t)n_dev);
    std::vector<float*> P0((size_t)n_dev), P1((size_t)n_dev), P2((size_t)n_dev);
    for (int32_t k = 0; k < n_dev; ++k) {
        DS& d = S[(size_t)k];
        const size_t nl = (size_t)(d.x1 - d.x0 + 4) * (size_t)sx;
        A0[k].assign(nl, 0.f); A1[k].assign(nl, 0.f); A2[k].assign(nl, 0.f);
        PSI[k].assign(d.pc.size(), 0.f); PHI[k].assign(d.pc.size(), 0.f);
        P0[k] = A0[k].data() - d.base; P1[k] = A1[k].data() - d.base; P2[k] = A2[k].data() - d.base;
    }
    for (int32_t it = 0; it < n_steps; ++it) {
        if (g_fdtd_progress) *g_fdtd_progress = it;
        for (int32_t k = 0; k < n_dev; ++k) {
            DS& d = S[(size_t)k];
            const int64_t t0 = d.x0 * ny * nz, n = (d.x1 - d.x0) * ny * nz;
            for (int64_t t = t0; t < t0 + n; ++t)
                dense_cell(B, t, c4c, c4nb, c4nb2, coef, P0[k], P1[k], P2[k]);
            for (int64_t q = 0; q < d.eoff[3]; ++q)
                dense_psi(q, d.pc.data(), d.pp.data(), d.pb.data(), d.pap.data(), P1[k], PSI[k].data());
            for (int a = 0; a < 3; ++a)
                for (int64_t q = 0; q < d.eoff[a + 1] - d.eoff[a]; ++q)
                    dense_apply(d.eoff[a], q, d.pc.data(), d.pm.data(), d.pp.data(), d.pl.data(),
                                d.pb.data(), d.pa.data(), PSI[k].data(), PHI[k].data(), lam2,
                                P1[k], P2[k]);
            if (it < n_pulse)
                for (size_t q = 0; q < d.si.size(); ++q) P2[k][d.si[q]] += pulse[it] * d.sw[q];
        }
        for (int32_t k = 0; k < n_dev; ++k) {
            DS& d = S[(size_t)k];
            if (k > 0)
                std::memcpy(P2[k] + (d.x0 + 0) * sx, P2[k - 1] + (d.x0 + 0) * sx, (size_t)(2 * sx) * 4);
            if (k + 1 < n_dev)
                std::memcpy(P2[k] + (d.x1 + 2) * sx, P2[k + 1] + (d.x1 + 2) * sx, (size_t)(2 * sx) * 4);
            for (size_t r = 0; r < d.recs.size(); ++r) {
                float acc = 0.f;
                for (int q = 0; q < 8; ++q) acc += P2[k][d.ri[8 * r + q]] * d.rw[8 * r + q];
                out[(size_t)it * n_rec + d.recs[r]] = acc;
            }
        }
        for (int32_t k = 0; k < n_dev; ++k) { float* t = P0[k]; P0[k] = P1[k]; P1[k] = P2[k]; P2[k] = t; }
    }
    return 0;
#endif
}

AUR_API const char* aur_version(void) {
#if AUR_GPU
    return "auralis-cuda 2.8.1";
#else
    return "auralis-cuda-emu 2.8.1";
#endif
}

} /* extern "C" */
