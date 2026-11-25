/* Auralis native core: CPU ray tracer and FDTD (build with native/build.sh).
 * The tracer's structure follows the Vercidium Audio model (vercidium.com). */
#include "aur_native.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#include <cstdio>

namespace {

/* Small math */
struct V3 {
    float x, y, z;
};
static inline V3 v3(float x, float y, float z) { return {x, y, z}; }
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline float len(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 norm(V3 a) {
    float l = len(a);
    return l > 0.f ? a * (1.f / l) : v3(0, 0, 1);
}
static inline V3 vmin(V3 a, V3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
static inline V3 vmax(V3 a, V3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
static inline float comp(V3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }

/* PCG32 */
struct Pcg32 {
    uint64_t state, inc;
    explicit Pcg32(uint64_t seed, uint64_t seq = 0xda3e39cb94b95bdbULL) {
        state = 0u; inc = (seq << 1u) | 1u;
        next(); state += seed; next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xs >> rot) | (xs << ((-rot) & 31));
    }
    float f01() { return (next() >> 8) * (1.0f / 16777216.0f); } /* [0,1) */
};

/* BVH */
struct AABB {
    V3 lo{ 1e30f,  1e30f,  1e30f};
    V3 hi{-1e30f, -1e30f, -1e30f};
    void grow(V3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void grow(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    float area() const {
        V3 d = hi - lo;
        if (d.x < 0 || d.y < 0 || d.z < 0) return 0.f;
        return 2.f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

struct BvhNode {              /* 32 bytes */
    V3 lo; int32_t left;      /* left child index, or first tri if leaf   */
    V3 hi; int32_t count;     /* 0 = interior, >0 = leaf triangle count   */
};

} /* anonymous namespace */

struct AurScene {
    /* SoA triangle data, in BVH order */
    std::vector<V3> v0, e1, e2, ng;      /* geometric normal (unit)        */
    std::vector<int32_t> mat;            /* material per (reordered) tri   */
    std::vector<int32_t> src_index;      /* reordered -> original index    */
    std::vector<BvhNode> nodes;
    int32_t n_tris = 0;
};

namespace {

constexpr int   SAH_BINS   = 12;
constexpr int   LEAF_MAX   = 4;
constexpr float RAY_EPS    = 1e-4f;

struct BuildTri { AABB box; V3 centroid; int32_t idx; };

/* Right-child links are kept in a side array so nodes stay 32 bytes. */
struct BvhLinks { std::vector<int32_t> right; };

static int32_t build2(AurScene& s, BvhLinks& L, std::vector<BuildTri>& tris,
                      int32_t begin, int32_t end, int depth) {
    AABB box, cbox;
    for (int32_t i = begin; i < end; ++i) { box.grow(tris[i].box); cbox.grow(tris[i].centroid); }
    int32_t node_idx = (int32_t)s.nodes.size();
    s.nodes.push_back({box.lo, 0, box.hi, 0});
    L.right.push_back(-1);
    int32_t count = end - begin;
    if (count <= LEAF_MAX || depth > 60) {
        s.nodes[node_idx].left = begin;
        s.nodes[node_idx].count = count;
        return node_idx;
    }
    V3 cext = cbox.hi - cbox.lo;
    int axis = 0;
    if (cext.y > comp(cext, axis)) axis = 1;
    if (cext.z > comp(cext, axis)) axis = 2;
    float cmin = comp(cbox.lo, axis), cext_a = comp(cext, axis);
    int32_t mid = -1;
    if (cext_a > 1e-9f) {
        AABB bin_box[SAH_BINS]; int bin_cnt[SAH_BINS] = {0};
        float k = SAH_BINS * (1.f - 1e-6f) / cext_a;
        for (int32_t i = begin; i < end; ++i) {
            int b = (int)((comp(tris[i].centroid, axis) - cmin) * k);
            b = std::min(std::max(b, 0), SAH_BINS - 1);
            bin_box[b].grow(tris[i].box); bin_cnt[b]++;
        }
        AABB right_acc[SAH_BINS]; AABB acc;
        for (int b = SAH_BINS - 1; b > 0; --b) { acc.grow(bin_box[b]); right_acc[b] = acc; }
        float best = 1e30f; int best_bin = -1; AABB lacc; int32_t running = 0;
        for (int b = 0; b < SAH_BINS - 1; ++b) {
            lacc.grow(bin_box[b]); running += bin_cnt[b];
            if (running == 0 || running == count) continue;
            float cost = lacc.area() * running + right_acc[b + 1].area() * (count - running);
            if (cost < best) { best = cost; best_bin = b; }
        }
        if (best_bin >= 0) {
            float split = cmin + (best_bin + 1) * cext_a / SAH_BINS;
            auto it = std::partition(tris.begin() + begin, tris.begin() + end,
                [&](const BuildTri& t) { return comp(t.centroid, axis) < split; });
            mid = (int32_t)(it - tris.begin());
            if (mid == begin || mid == end) mid = -1;
        }
    }
    if (mid < 0) {
        mid = begin + count / 2;
        std::nth_element(tris.begin() + begin, tris.begin() + mid, tris.begin() + end,
            [&](const BuildTri& a, const BuildTri& b) {
                return comp(a.centroid, axis) < comp(b.centroid, axis);
            });
    }
    int32_t l = build2(s, L, tris, begin, mid, depth + 1);
    int32_t r = build2(s, L, tris, mid, end, depth + 1);
    s.nodes[node_idx].left = l;
    L.right[node_idx] = r;
    return node_idx;
}

/* Slab test */
static inline bool hit_aabb(const BvhNode& n, V3 o, V3 inv_d, float t_max) {
    float tx1 = (n.lo.x - o.x) * inv_d.x, tx2 = (n.hi.x - o.x) * inv_d.x;
    float tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
    float ty1 = (n.lo.y - o.y) * inv_d.y, ty2 = (n.hi.y - o.y) * inv_d.y;
    tmin = std::max(tmin, std::min(ty1, ty2)); tmax = std::min(tmax, std::max(ty1, ty2));
    float tz1 = (n.lo.z - o.z) * inv_d.z, tz2 = (n.hi.z - o.z) * inv_d.z;
    tmin = std::max(tmin, std::min(tz1, tz2)); tmax = std::min(tmax, std::max(tz1, tz2));
    return tmax >= std::max(tmin, 0.0f) && tmin < t_max;
}

/* Moller-Trumbore against precomputed v0/e1/e2 */
static inline bool hit_tri(const AurScene& s, int32_t i, V3 o, V3 d,
                           float t_max, float& t_out) {
    V3 p = cross(d, s.e2[i]);
    float det = dot(s.e1[i], p);
    if (std::fabs(det) < 1e-12f) return false;
    float inv = 1.f / det;
    V3 tv = o - s.v0[i];
    float u = dot(tv, p) * inv;
    if (u < -1e-6f || u > 1.f + 1e-6f) return false;
    V3 q = cross(tv, s.e1[i]);
    float v = dot(d, q) * inv;
    if (v < -1e-6f || u + v > 1.f + 1e-6f) return false;
    float t = dot(s.e2[i], q) * inv;
    if (t <= RAY_EPS || t >= t_max) return false;
    t_out = t;
    return true;
}

static thread_local int32_t g_stack[128];

static int32_t trace_nearest(const AurScene& s, const int32_t* right,
                             V3 o, V3 d, float t_max, float& t_hit) {
    V3 inv_d = { 1.f / (d.x == 0.f ? 1e-30f : d.x),
                 1.f / (d.y == 0.f ? 1e-30f : d.y),
                 1.f / (d.z == 0.f ? 1e-30f : d.z) };
    int sp = 0; g_stack[sp++] = 0;
    int32_t best = -1; t_hit = t_max;
    while (sp) {
        int32_t ni = g_stack[--sp];
        const BvhNode& n = s.nodes[ni];
        if (!hit_aabb(n, o, inv_d, t_hit)) continue;
        if (n.count) {
            for (int32_t i = n.left; i < n.left + n.count; ++i) {
                float t;
                if (hit_tri(s, i, o, d, t_hit, t)) { t_hit = t; best = i; }
            }
        } else {
            g_stack[sp++] = right[ni];
            g_stack[sp++] = n.left;
        }
    }
    return best;
}

/* any hit (occlusion) */
static bool trace_any(const AurScene& s, const int32_t* right,
                      V3 o, V3 d, float t_max) {
    V3 inv_d = { 1.f / (d.x == 0.f ? 1e-30f : d.x),
                 1.f / (d.y == 0.f ? 1e-30f : d.y),
                 1.f / (d.z == 0.f ? 1e-30f : d.z) };
    int sp = 0; g_stack[sp++] = 0;
    while (sp) {
        int32_t ni = g_stack[--sp];
        const BvhNode& n = s.nodes[ni];
        if (!hit_aabb(n, o, inv_d, t_max)) continue;
        if (n.count) {
            for (int32_t i = n.left; i < n.left + n.count; ++i) {
                float t;
                if (hit_tri(s, i, o, d, t_max, t)) return true;
            }
        } else {
            g_stack[sp++] = right[ni];
            g_stack[sp++] = n.left;
        }
    }
    return false;
}

/* cosine-weighted hemisphere around n (unit) */
static inline V3 sample_lambert(Pcg32& rng, V3 n) {
    float u1 = rng.f01(), u2 = rng.f01();
    float r = std::sqrt(u1), phi = 6.28318530718f * u2;
    float x = r * std::cos(phi), y = r * std::sin(phi);
    float z = std::sqrt(std::max(0.f, 1.f - u1));
    V3 a = std::fabs(n.x) > 0.9f ? v3(0, 1, 0) : v3(1, 0, 0);
    V3 t = norm(cross(a, n));
    V3 b = cross(n, t);
    return norm(t * x + b * y + n * z);
}

static inline V3 sample_sphere(Pcg32& rng) {
    float z = 1.f - 2.f * rng.f01();
    float r = std::sqrt(std::max(0.f, 1.f - z * z));
    float phi = 6.28318530718f * rng.f01();
    return {r * std::cos(phi), r * std::sin(phi), z};
}

/* Gaussian pair (Box-Muller): anti-crack re-aim of the tracer */
static inline void gauss2(Pcg32& rng, float& g1, float& g2) {
    float u1 = rng.f01(); if (u1 < 1e-12f) u1 = 1e-12f;
    float u2 = rng.f01();
    float r = std::sqrt(-2.f * std::log(u1));
    g1 = r * std::cos(6.28318530718f * u2);
    g2 = r * std::sin(6.28318530718f * u2);
}

} /* anonymous namespace */

/* C API */

struct AurSceneWrapper {
    AurScene scene;
    std::vector<int32_t> right;
};

extern "C" {

AUR_API AurScene* aur_scene_create(const float* verts, int32_t n_tris,
                                   const int32_t* mat_idx) {
    if (!verts || n_tris <= 0) return nullptr;
    auto* w = new AurSceneWrapper();
    AurScene& s = w->scene;
    s.n_tris = n_tris;

    std::vector<BuildTri> btris((size_t)n_tris);
    for (int32_t i = 0; i < n_tris; ++i) {
        const float* p = verts + 9 * (size_t)i;
        V3 a = v3(p[0], p[1], p[2]);
        V3 b = v3(p[3], p[4], p[5]);
        V3 c = v3(p[6], p[7], p[8]);
        AABB box; box.grow(a); box.grow(b); box.grow(c);
        btris[i] = {box, (a + b + c) * (1.f / 3.f), i};
    }
    s.nodes.reserve((size_t)n_tris * 2);
    BvhLinks L; L.right.reserve((size_t)n_tris * 2);
    build2(s, L, btris, 0, n_tris, 0);
    w->right = std::move(L.right);

    s.v0.resize(n_tris); s.e1.resize(n_tris); s.e2.resize(n_tris);
    s.ng.resize(n_tris); s.mat.resize(n_tris); s.src_index.resize(n_tris);
    for (int32_t i = 0; i < n_tris; ++i) {
        int32_t oi = btris[i].idx;
        const float* p = verts + 9 * (size_t)oi;
        V3 a = v3(p[0], p[1], p[2]);
        V3 b = v3(p[3], p[4], p[5]);
        V3 c = v3(p[6], p[7], p[8]);
        s.v0[i] = a; s.e1[i] = b - a; s.e2[i] = c - a;
        s.ng[i] = norm(cross(s.e1[i], s.e2[i]));
        s.mat[i] = mat_idx ? mat_idx[oi] : 0;
        s.src_index[i] = oi;
    }
    return &w->scene;   /* AurScene is first member: pointer-compatible */
}

AUR_API void aur_scene_destroy(AurScene* scene) {
    if (!scene) return;
    delete reinterpret_cast<AurSceneWrapper*>(scene);
}

AUR_API int32_t aur_scene_node_count(const AurScene* scene) {
    return scene ? (int32_t)scene->nodes.size() : 0;
}

AUR_API int32_t aur_raycast(const AurScene* scene, const float o[3],
                            const float d[3], float t_max,
                            float* t_out, float n_out[3]) {
    if (!scene) return -1;
    const auto* w = reinterpret_cast<const AurSceneWrapper*>(scene);
    float t;
    int32_t i = trace_nearest(*scene, w->right.data(),
                              v3(o[0], o[1], o[2]), v3(d[0], d[1], d[2]),
                              t_max, t);
    if (i < 0) return -1;
    if (t_out) *t_out = t;
    if (n_out) { n_out[0] = scene->ng[i].x; n_out[1] = scene->ng[i].y; n_out[2] = scene->ng[i].z; }
    return scene->src_index[i];
}

AUR_API int32_t aur_segment_transmission(const AurScene* scene,
                                         const float a[3], const float b[3],
                                         const float* transmission,
                                         int32_t n_mats, int32_t n_bands,
                                         float* out_trans) {
    for (int32_t k = 0; k < n_bands; ++k) out_trans[k] = 1.f;
    if (!scene) return 0;
    const auto* w = reinterpret_cast<const AurSceneWrapper*>(scene);
    V3 A = v3(a[0], a[1], a[2]), B = v3(b[0], b[1], b[2]);
    V3 d = B - A;
    float dist = len(d);
    if (dist < 1e-6f) return 0;
    d = d * (1.f / dist);
    V3 o = A;
    float remaining = dist - RAY_EPS;
    int32_t crossings = 0;
    for (int guard = 0; guard < 256 && remaining > RAY_EPS; ++guard) {
        float t;
        int32_t i = trace_nearest(*scene, w->right.data(), o, d, remaining, t);
        if (i < 0) break;
        int32_t m = std::min(std::max(scene->mat[i], 0), n_mats - 1);
        const float* tr = transmission + (size_t)m * n_bands;
        for (int32_t k = 0; k < n_bands; ++k) out_trans[k] *= tr[k];
        crossings++;
        o = o + d * (t + RAY_EPS);
        remaining -= (t + RAY_EPS);
        float mx = 0.f;
        for (int32_t k = 0; k < n_bands; ++k) mx = std::max(mx, out_trans[k]);
        if (mx < 1e-7f) break;
    }
    return crossings;
}

/* reflection-based scattering (Christensen & Rindel 2005, eqs. 1-5) */
static inline float rbs_sd(float f, float l, float w, float cos_i, float astar,
                           float c) {
    float ws = std::max(w * cos_i, 1e-3f), ls = std::max(l, 1e-3f);
    float a = std::max(astar, 1e-6f);
    float kw = std::min(1.f, f / (c * a / (2.f * ws * ws)));
    float kl = std::min(1.f, f / (c * a / (2.f * ls * ls)));
    return 1.f - kw * kl;
}
static inline float rbs_combine(float sd, float ss) {
    return 1.f - (1.f - sd) * (1.f - ss);
}
static inline float rbs_astar(float di, float dr) {
    return di * dr / std::max(2.f * (di + dr), 1e-12f);
}

static int32_t trace_impl(const AurScene* scene, const AurTraceParams* p,
                          const AurRbs* rbs,
                          float* out_hist, float* out_dir, double* out_stats) {
    if (!scene || !p || !out_hist || !out_dir) return 1;
    if (rbs && (!rbs->tri_l || !rbs->tri_w || !rbs->band_hz)) return 3;
    if (p->n_bands <= 0 || p->n_bands > 16) return 2;
    const auto* w = reinterpret_cast<const AurSceneWrapper*>(scene);
    const int32_t* right = w->right.data();

    const int NB = p->n_bands;
    const int NBINS = p->n_bins;
    const float c = p->speed_of_sound;
    const float inv_bin = 1.f / p->bin_dt;
    const int K = std::max(0, p->spec_exclude_order);
    const float cap = std::max(1e-9f, p->energy_cap);
    const double inv_n = 1.0 / std::max(1, p->n_rays);
    /* sphere detection weight 4/r^2: expectation = image lattice (1/d^2) */
    const float r_det = std::max(1e-3f, p->rcv_radius);
    const float r_det2 = r_det * r_det;
    const double det_w = 4.0 / ((double)r_det * (double)r_det);

    /* per-band air attenuation: energy factor exp(-k * meters), k = ln(10)/10 * dB/m */
    float kair[16];
    for (int b = 0; b < NB; ++b)
        kair[b] = 0.2302585093f * (p->air_db_per_m ? p->air_db_per_m[b] : 0.f);

    int n_threads = p->n_threads > 0 ? p->n_threads
                                     : (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 128);

    std::vector<std::vector<double>> hists((size_t)n_threads),
                                     dirs((size_t)n_threads);
    std::vector<double> t_bounces((size_t)n_threads, 0.0),
                        t_escaped((size_t)n_threads, 0.0),
                        t_deposits((size_t)n_threads, 0.0);

    V3 SRC = v3(p->src[0], p->src[1], p->src[2]);
    V3 RCV = v3(p->rcv[0], p->rcv[1], p->rcv[2]);
    V3 FWD = v3(p->src_forward[0], p->src_forward[1], p->src_forward[2]);
    float fwd_len = std::sqrt(dot(FWD, FWD));
    const bool has_dir = (p->n_dir_samples > 1) && (p->dir_table != nullptr)
                         && (fwd_len > 1e-6f);
    if (has_dir) FWD = FWD * (1.0f / fwd_len);
    const int NDIR = has_dir ? p->n_dir_samples : 0;

    /* one double histogram per thread, merged at the end: no shared mutable state */
    auto worker = [&](int tid, int32_t r0, int32_t r1) {
        auto& H = hists[tid]; H.assign((size_t)NB * NBINS, 0.0);
        auto& D = dirs[tid];  D.assign((size_t)NBINS * 3, 0.0);
        double bounces = 0.0, escaped = 0.0, deposits = 0.0;
        Pcg32 rng(p->seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1), 77 + tid);

        float E[16];
        for (int32_t ri = r0; ri < r1; ++ri) {
            V3 o = SRC;
            V3 d = sample_sphere(rng);
            if (has_dir) {
                float ct = dot(d, FWD);
                if (ct > 1.f) ct = 1.f; else if (ct < -1.f) ct = -1.f;
                float u = std::acos(ct) * (float)(NDIR - 1)
                          / 3.14159265358979f;
                int i0 = (int)u;
                if (i0 > NDIR - 2) i0 = NDIR - 2;
                if (i0 < 0) i0 = 0;
                float fr = u - (float)i0;
                for (int b = 0; b < NB; ++b) {
                    const float* row = p->dir_table + (size_t)b * NDIR;
                    E[b] = row[i0] * (1.f - fr) + row[i0 + 1] * fr;
                }
            } else {
                for (int b = 0; b < NB; ++b) E[b] = 1.f;
            }
            float time = 0.f;
            int   order = 0;          /* completed reflections            */
            bool  pure_spec = true;
            bool  spec_seg = true;    /* segment off a specular branch    */
            int   retries = 0;        /* anti-crack re-aim (launch incl.) */
            float dspec = 0.f;        /* path since the last diffuse event */
            float wsp[16];            /* rbs: unbiased per-band weight of a
                                         specular segment, (1 - s_b) / (1 - s_mean) */
            for (int b = 0; b < NB; ++b) wsp[b] = 1.f;

            for (int bounce = 0; bounce < p->max_bounces; ++bounce) {
                float t_hit;
                int32_t tri = trace_nearest(*scene, right, o, d, 1e30f, t_hit);
                bool miss = (tri < 0);

                /* numerical cracks: re-aim before giving up */
                if (miss && retries < 2) {
                    ++retries;
                    float g1, g2, g3, g4;
                    gauss2(rng, g1, g2);
                    gauss2(rng, g3, g4);
                    V3 dj = v3(d.x + 0.008f * g1, d.y + 0.008f * g2,
                               d.z + 0.008f * g3);
                    d = norm(dj);
                    continue;         /* costs one bounce, as in python   */
                }

                /* sphere detection of the specular energy, final segments only (hit or
                 * confirmed escape); pure specular chains of order <= K (order 0 = the
                 * direct sound) are left to the image sources */
                float segl = miss ? std::max(0.f, c * (p->max_time - time))
                                  : t_hit;
                if (segl > 0.f && spec_seg && !(pure_spec && order <= K)) {
                    V3 wv = RCV - o;
                    float tc = dot(wv, d);
                    if (tc > 0.f && tc < segl) {
                        float m2 = dot(wv, wv) - tc * tc;
                        if (m2 < r_det2) {
                            int bin = (int)((time + tc / c) * inv_bin);
                            if (bin >= 0 && bin < NBINS) {
                                double wsum = 0.0;
                                for (int b = 0; b < NB; ++b) {
                                    double dep = (double)E[b]
                                        * std::exp(-kair[b] * tc)
                                        * det_w * inv_n;
                                    if (rbs) dep *= wsp[b];
                                    H[(size_t)b * NBINS + bin] += dep;
                                    wsum += dep;
                                }
                                /* DOA: flies along d, arrives from -d   */
                                D[(size_t)bin * 3 + 0] += -wsum * d.x;
                                D[(size_t)bin * 3 + 1] += -wsum * d.y;
                                D[(size_t)bin * 3 + 2] += -wsum * d.z;
                                deposits += 1.0;
                            }
                        }
                    }
                }
                if (miss) { escaped += 1.0; break; }

                for (int b = 0; b < NB; ++b) E[b] *= std::exp(-kair[b] * t_hit);
                time += t_hit / c;
                if (time > p->max_time) break;

                V3 hp = o + d * t_hit;
                V3 n  = scene->ng[tri];
                if (dot(n, d) > 0.f) n = n * -1.f;       /* face the ray   */

                int32_t m = scene->mat[tri];
                if (m < 0) { m = 0; }
                if (m >= p->n_mats) { m = p->n_mats - 1; }
                const float* alpha = p->absorption   + (size_t)m * NB;
                const float* scat  = p->scattering   + (size_t)m * NB;
                const float* tau   = p->transmission + (size_t)m * NB;

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
                if (rbs) {                /* the ray: reflected = incident */
                    d_in = dspec + t_hit;
                    cos_i = std::fabs(dot(n, d));
                    const int32_t ot = scene->src_index[tri];  /* BVH order */
                    tl = rbs->tri_l[ot];
                    tw = rbs->tri_w[ot];
                    float sm = 0.f;
                    for (int b = 0; b < NB; ++b) {
                        s_c[b] = rbs_combine(rbs_sd(rbs->band_hz[b], tl, tw, cos_i,
                                                    0.25f * d_in, c), scat[b]);
                        sm += s_c[b];
                    }
                    s_mean = sm / NB;
                }

                /* diffuse rain toward the receiver: only the scattered part s*E radiates
                 * (Lambert); the specular part is caught by the sphere detection */
                V3 to_r = RCV - hp;
                float dR = len(to_r);
                if (dR > 1e-4f) {
                    V3 ur = to_r * (1.f / dR);
                    float costh = dot(n, ur);
                    if (costh > 0.f) {
                        int bin = (int)((time + dR / c) * inv_bin);
                        if (bin >= 0 && bin < NBINS) {
                            V3 off = hp + n * (RAY_EPS * 4.f);
                            if (!trace_any(*scene, right, off, ur, dR - RAY_EPS * 8.f)) {
                                double wsum = 0.0;
                                float geo = 4.f * costh / (dR * dR);
                                for (int b = 0; b < NB; ++b) {
                                    float sv = scat[b];
                                    if (rbs)  /* the last leg's true length */
                                        sv = rbs_combine(rbs_sd(rbs->band_hz[b], tl, tw,
                                                                cos_i, rbs_astar(d_in, dR), c),
                                                         scat[b]);
                                    double dep = (double)Eref[b] * sv * geo
                                               * std::exp(-kair[b] * dR) * inv_n;
                                    H[(size_t)b * NBINS + bin] += dep;
                                    wsum += dep;
                                }
                                /* DOA: from receiver toward the wall     */
                                D[(size_t)bin * 3 + 0] += wsum * (hp.x - RCV.x) / dR;
                                D[(size_t)bin * 3 + 1] += wsum * (hp.y - RCV.y) / dR;
                                D[(size_t)bin * 3 + 2] += wsum * (hp.z - RCV.z) / dR;
                                deposits += 1.0;
                            }
                        }
                    }
                }

                bounces += 1.0;
                order += 1;
                for (int b = 0; b < NB; ++b) E[b] = Eref[b];

                /* unbiased Russian roulette below the energy cap */
                if (e_max < cap) {
                    const float p_surv = 0.25f;
                    if (rng.f01() > p_surv) break;
                    for (int b = 0; b < NB; ++b) E[b] *= (1.f / p_surv);
                    if (e_max < cap * 1e-4f) break;   /* hard floor       */
                }

                bool go_diff = (rng.f01() < s_mean);
                if (go_diff) {
                    d = sample_lambert(rng, n);
                    pure_spec = false;
                } else {
                    d = d - n * (2.f * dot(d, n));
                }
                spec_seg = !go_diff;
                if (rbs) {
                    dspec = go_diff ? 0.f : d_in;
                    if (!go_diff) {
                        float q = 1.f / std::max(1.f - s_mean, 1e-6f);
                        for (int b = 0; b < NB; ++b) wsp[b] = (1.f - s_c[b]) * q;
                    }
                }
                o = hp + n * (RAY_EPS * 4.f);
            }
        }
        t_bounces[tid] = bounces;
        t_escaped[tid] = escaped;
        t_deposits[tid] = deposits;
    };

    if (n_threads == 1) {
        worker(0, 0, p->n_rays);
    } else {
        std::vector<std::thread> pool;
        int32_t chunk = (p->n_rays + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; ++t) {
            int32_t r0 = t * chunk, r1 = std::min(p->n_rays, r0 + chunk);
            if (r0 >= r1) break;
            pool.emplace_back(worker, t, r0, r1);
        }
        for (auto& th : pool) th.join();
    }

    std::memset(out_hist, 0, sizeof(float) * (size_t)NB * NBINS);
    std::memset(out_dir,  0, sizeof(float) * (size_t)NBINS * 3);
    double tb = 0, te = 0, td = 0;
    for (int t = 0; t < n_threads; ++t) {
        if (hists[t].empty()) continue;
        for (size_t i = 0; i < (size_t)NB * NBINS; ++i) out_hist[i] += (float)hists[t][i];
        for (size_t i = 0; i < (size_t)NBINS * 3;  ++i) out_dir[i]  += (float)dirs[t][i];
        tb += t_bounces[t]; te += t_escaped[t]; td += t_deposits[t];
    }
    if (out_stats) {
        out_stats[0] = (double)p->n_rays;
        out_stats[1] = tb / std::max(1, p->n_rays);
        out_stats[2] = te;
        out_stats[3] = td;
    }
    return 0;
}


AUR_API int32_t aur_trace(const AurScene* scene, const AurTraceParams* p,
                          float* out_hist, float* out_dir, double* out_stats) {
    return trace_impl(scene, p, nullptr, out_hist, out_dir, out_stats);
}

AUR_API int32_t aur_trace2(const AurScene* scene, const AurTraceParams* p,
                           const AurRbs* rbs, float* out_hist, float* out_dir,
                           double* out_stats) {
    return trace_impl(scene, p, rbs, out_hist, out_dir, out_stats);
}



/* FDTD progress: the step being computed, for a host poller */
static volatile int64_t* g_fdtd_progress = nullptr;
extern "C" AUR_API void aur_fdtd_progress(int64_t* counter) {
    g_fdtd_progress = counter;
}
AUR_API int32_t aur_fdtd_run(const int32_t* nb,
                             const float* c_now, const float* c_old,
                             const float* c_nb, int64_t n_air,
                             const float* pulse, int32_t n_pulse,
                             const int32_t* src_idx, const float* src_w,
                             int32_t n_src,
                             const int32_t* rec_idx, const float* rec_w,
                             int32_t n_rec,
                             int32_t n_steps, int32_t n_threads,
                             float* out) {
    if (!nb || !c_now || n_air <= 0 || n_steps <= 0 || !out) return 1;
    std::vector<float> P0((size_t)n_air + 1, 0.f), P1((size_t)n_air + 1, 0.f),
                       P2((size_t)n_air + 1, 0.f);
    float* p0 = P0.data(); float* p1 = P1.data(); float* p2 = P2.data();

    int nt = n_threads > 0 ? n_threads
             : (int)std::max(1u, std::thread::hardware_concurrency());
    nt = std::min<int>(nt, 64);
    int64_t chunk = (n_air + nt - 1) / nt;

    auto cell_range = [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) {
            const int32_t* q = nb + 6 * i;
            float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                    + p1[q[3]] + p1[q[4]] + p1[q[5]];
            p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i];
        }
    };

    for (int32_t it = 0; it < n_steps; ++it) {

        if (g_fdtd_progress) *g_fdtd_progress = it;
        if (nt == 1 || n_air < 20000) {
            cell_range(0, n_air);
        } else {
            std::vector<std::thread> pool;
            for (int t = 0; t < nt; ++t) {
                int64_t a = t * chunk, b = std::min(n_air, a + chunk);
                if (a >= b) break;
                pool.emplace_back(cell_range, a, b);
            }
            for (auto& th : pool) th.join();
        }
        p2[n_air] = 0.f;
        if (it < n_pulse) {
            for (int32_t k = 0; k < n_src; ++k)
                p2[src_idx[k]] += pulse[it] * src_w[k];
        }
        for (int32_t r = 0; r < n_rec; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 8; ++k)
                acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
            out[(size_t)it * n_rec + r] = acc;
        }
        float* tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
    }
    return 0;
}


AUR_API int32_t aur_fdtd_run2(const int32_t* nb,
                              const float* c_now, const float* c_old,
                              const float* c_nb, const float* c_drv,
                              int64_t n_air,
                              int64_t n_faces, const int32_t* f_cell,
                              const float* f_e, const float* f_g,
                              const float* f_l, const float* f_Asum,
                              int64_t n_rfaces, const int32_t* r_cell,
                              const float* r_A, const float* r_c1,
                              const float* r_c2, const float* r_E,
                              const float* r_G,
                              const float* pulse, int32_t n_pulse,
                              const int32_t* src_idx, const float* src_w,
                              int32_t n_src,
                              const int32_t* rec_idx, const float* rec_w,
                              int32_t n_rec,
                              int32_t n_steps, int32_t n_threads,
                              float* out) {
    /* filtering walls (2 poles per face, ADE) and resonant sections: drive
       rA p - (rc1 psi1 + rc2 psi2), psi propagated exactly (r_E 2x2, r_G) from (p2+p1)/2.
       Per step: drive D(p^n) -> stencil (- c_drv*D) -> source/probes -> phi, psi from p^{n+1}. */
    if (!nb || !c_now || n_air <= 0 || n_steps <= 0 || !out) return 1;
    std::vector<float> P0((size_t)n_air + 1, 0.f), P1((size_t)n_air + 1, 0.f),
                       P2((size_t)n_air + 1, 0.f);
    std::vector<float> D((size_t)n_air, 0.f);
    std::vector<float> phi(2 * (size_t)n_faces, 0.f);
    const bool have_r = (n_rfaces > 0 && r_cell);
    std::vector<float> psiR(have_r ? 2 * (size_t)n_rfaces : 0, 0.f);
    float* p0 = P0.data(); float* p1 = P1.data(); float* p2 = P2.data();

    int nt = n_threads > 0 ? n_threads
             : (int)std::max(1u, std::thread::hardware_concurrency());
    nt = std::min<int>(nt, 64);
    int64_t chunk = (n_air + nt - 1) / nt;

    auto cell_range = [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) {
            const int32_t* q = nb + 6 * i;
            float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                    + p1[q[3]] + p1[q[4]] + p1[q[5]];
            p2[i] = c_now[i] * p1[i] + c_nb[i] * s + c_old[i] * p0[i]
                    - c_drv[i] * D[i];
        }
    };

    for (int32_t it = 0; it < n_steps; ++it) {

        if (g_fdtd_progress) *g_fdtd_progress = it;
        /* faces sequential: cheap, and no write collisions on corner cells */
        std::fill(D.begin(), D.end(), 0.f);
        for (int64_t f = 0; f < n_faces; ++f) {
            const int64_t c = f_cell[f];
            D[c] += f_Asum[f] * p1[c]
                  - (f_l[2 * f] * phi[2 * f]
                     + f_l[2 * f + 1] * phi[2 * f + 1]);
        }
        if (have_r)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                D[c] += r_A[f] * p1[c]
                      - (r_c1[f] * psiR[2 * f]
                         + r_c2[f] * psiR[2 * f + 1]);
            }
        if (nt == 1 || n_air < 20000) {
            cell_range(0, n_air);
        } else {
            std::vector<std::thread> pool;
            for (int t = 0; t < nt; ++t) {
                int64_t a = t * chunk, b = std::min(n_air, a + chunk);
                if (a >= b) break;
                pool.emplace_back(cell_range, a, b);
            }
            for (auto& th : pool) th.join();
        }
        p2[n_air] = 0.f;
        if (it < n_pulse) {
            for (int32_t k = 0; k < n_src; ++k)
                p2[src_idx[k]] += pulse[it] * src_w[k];
        }
        for (int32_t r = 0; r < n_rec; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 8; ++k)
                acc += p2[rec_idx[8 * r + k]] * rec_w[8 * r + k];
            out[(size_t)it * n_rec + r] = acc;
        }
        for (int64_t f = 0; f < n_faces; ++f) {
            const float pc = 0.5f * (p2[f_cell[f]] + p1[f_cell[f]]);
            phi[2 * f]     = f_e[2 * f]     * phi[2 * f]     + f_g[2 * f]     * pc;
            phi[2 * f + 1] = f_e[2 * f + 1] * phi[2 * f + 1] + f_g[2 * f + 1] * pc;
        }
        if (have_r)
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
}

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
    /* 4th-order interior (13-point stencil, mask4), order 2 at the boundary, per-axis CPML.
       Step order IDENTICAL to run_grid_python: drive D (p1) -> stencil -> links -> CPML (psi
       all axes, then phi/apply) -> ghost -> source -> probes -> wall phi (p2) -> rotation. */
    if (!nb || !nb2 || !mask4 || !c_now || n_air <= 0 || n_steps <= 0
        || !out) return 1;
    std::vector<float> P0((size_t)n_air + 1, 0.f),
                       P1((size_t)n_air + 1, 0.f),
                       P2((size_t)n_air + 1, 0.f);
    float* p0 = P0.data(); float* p1 = P1.data(); float* p2 = P2.data();
    std::vector<float> D;
    std::vector<float> phiF;
    const bool have_r4 = (n_rfaces > 0 && r_cell);
    std::vector<float> psiR;
    if (n_faces > 0 || have_r4)
        D.assign((size_t)n_air, 0.f);
    if (n_faces > 0)
        phiF.assign(2 * (size_t)n_faces, 0.f);
    if (have_r4)
        psiR.assign(2 * (size_t)n_rfaces, 0.f);
    int64_t off[4] = {0, 0, 0, 0};
    int64_t n_pml = 0;
    if (pml_counts) {
        for (int a = 0; a < 3; ++a) {
            off[a] = n_pml;
            n_pml += pml_counts[a];
        }
        off[3] = n_pml;
    }
    std::vector<float> psi((size_t)n_pml, 0.f), phiP((size_t)n_pml, 0.f);

    int nt = n_threads > 0 ? n_threads
             : (int)std::max(1u, std::thread::hardware_concurrency());
    nt = std::min<int>(nt, 64);
    int64_t chunk = (n_air + nt - 1) / nt;
    const bool have_f = (n_faces > 0 && f_cell && c_drv);
    const bool have_d = (have_f || (have_r4 && c_drv));

    auto cell_range = [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) {
            if (mask4[i]) {
                const int32_t* q = nb + 6 * i;
                const int32_t* q2 = nb2 + 6 * i;
                float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                        + p1[q[3]] + p1[q[4]] + p1[q[5]];
                float s2 = p1[q2[0]] + p1[q2[1]] + p1[q2[2]]
                         + p1[q2[3]] + p1[q2[4]] + p1[q2[5]];
                p2[i] = c4c * p1[i] + c4nb * s + c4nb2 * s2 - p0[i];
            } else {
                const int32_t* q = nb + 6 * i;
                float s = p1[q[0]] + p1[q[1]] + p1[q[2]]
                        + p1[q[3]] + p1[q[4]] + p1[q[5]];
                p2[i] = c_now[i] * p1[i] + c_nb[i] * s
                      + c_old[i] * p0[i];
                if (have_d) p2[i] -= c_drv[i] * D[i];
            }
        }
    };

    const int sent_every = 512;
    double sent_ref = 0.0;
    /* optional biharmonic filter against GKS boundary modes, AUR_FDTD_FILTER="K,theta"
       (every K steps). OFF by default: it excites the mode more than it damps it. */
    int flt_every = 0;
    float flt_theta = 0.004f;
    { const char* fv = std::getenv("AUR_FDTD_FILTER");
      if (fv) { int k = 0; float th = 0.f;
                if (std::sscanf(fv, "%d,%f", &k, &th) == 2 &&
                    k > 0 && th > 0.f) { flt_every = k;
                                         flt_theta = th; }
                else flt_every = 0; } }
    std::vector<float> flt_q;
    if (flt_every > 0) flt_q.assign((size_t)n_air + 1, 0.f);
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
        if (have_r4)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                D[c] += r_A[f] * p1[c]
                      - (r_c1[f] * psiR[2 * f]
                         + r_c2[f] * psiR[2 * f + 1]);
            }
        if (nt == 1 || n_air < 20000) {
            cell_range(0, n_air);
        } else {
            std::vector<std::thread> pool;
            for (int t = 0; t < nt; ++t) {
                int64_t a = t * chunk, b = std::min(n_air, a + chunk);
                if (a >= b) break;
                pool.emplace_back(cell_range, a, b);
            }
            for (auto& th : pool) th.join();
        }
        /* conformal tangential apertures, CSR by cell: p2[i] += sum_k l_k (p1[nb] - p1[i]) */
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
                    phiP[k] = pml_b[k] * phiP[k]
                            + pml_a[k] * (d2 + dpsi);
                    p2[cc] += lam2 * (dpsi + phiP[k]);
                }
            }
        }
        p2[n_air] = 0.f;
        if (it < n_pulse) {
            for (int32_t k = 0; k < n_src; ++k)
                p2[src_idx[k]] += pulse[it] * src_w[k];
        }
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
        if (have_r4)
            for (int64_t f = 0; f < n_rfaces; ++f) {
                const int64_t c = r_cell[f];
                const float pc = 0.5f * (p2[c] + p1[c]);
                const float s1 = psiR[2 * f], s2 = psiR[2 * f + 1];
                psiR[2 * f]     = r_E[4 * f]     * s1 + r_E[4 * f + 1] * s2
                                + r_G[2 * f]     * pc;
                psiR[2 * f + 1] = r_E[4 * f + 2] * s1 + r_E[4 * f + 3] * s2
                                + r_G[2 * f + 1] * pc;
            }
        if (flt_every > 0 && (it % flt_every) == 0 && it > 0) {
            float* q = flt_q.data();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n_air; ++i) {
                const int32_t* v = nb + 6 * i;
                float acc = 0.f;
                for (int a = 0; a < 6; ++a)
                    if (v[a] != n_air)
                        acc += p1[v[a]] - p1[i];
                q[i] = acc;
            }
            q[n_air] = 0.f;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n_air; ++i) {
                const int32_t* v = nb + 6 * i;
                float lq = 0.f;
                for (int a = 0; a < 6; ++a)
                    if (v[a] != n_air)
                        lq += q[v[a]] - q[i];
                p1[i] -= flt_theta * lq;
            }
            /* p0 too: filtering one leapfrog level excites its spurious mode */
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n_air; ++i) {
                const int32_t* v = nb + 6 * i;
                float acc = 0.f;
                for (int a = 0; a < 6; ++a)
                    if (v[a] != n_air)
                        acc += p0[v[a]] - p0[i];
                q[i] = acc;
            }
            q[n_air] = 0.f;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n_air; ++i) {
                const int32_t* v = nb + 6 * i;
                float lq = 0.f;
                for (int a = 0; a < 6; ++a)
                    if (v[a] != n_air)
                        lq += q[v[a]] - q[i];
                p0[i] -= flt_theta * lq;
            }
        }
        if ((it % sent_every) == 0 && it > 0) {
            const int64_t stride = n_air / 25000 + 1;
            double mx = 0.0; bool bad = false;
            for (int64_t q = 0; q < n_air; q += stride) {
                double v = std::fabs((double)p1[q]);
                if (!(v == v)) { bad = true; break; }
                if (v > mx) mx = v;
            }
            if (sent_ref <= 0.0 && mx > 0.0) sent_ref = mx;
            if (bad || (sent_ref > 0.0 && mx > 1e4 * sent_ref)) {
                std::fprintf(stderr,
                    "[aur_fdtd] instability detected at step %d "
                    "(max|p| %.3e, ref %.3e)\n", it, mx,
                    sent_ref);
                return 41;
            }
        }
        float* tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
    }
    return 0;
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

/* run4 + sparse links (conformal tangential apertures). Both return 41 on instability. */
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

AUR_API const char* aur_last_gpu_error(void) { return ""; }

AUR_API int32_t aur_dev_ok(void* handle) { return handle ? 1 : 0; }

AUR_API const char* aur_version(void) { return "auralis-native 2.6.0"; }

} /* extern "C" */
