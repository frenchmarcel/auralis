/* Auralis native core: C ABI shared by libauralis and the CUDA libraries. */
#ifndef AUR_NATIVE_H
#define AUR_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
  #define AUR_API __declspec(dllexport)
#else
  #define AUR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque scene handle (triangle soup + BVH). One scene may serve several aur_trace calls in
 * sequence; each call spawns its own worker threads. */
typedef struct AurScene AurScene;

/* verts: 9 * n_tris floats (v0.xyz v1.xyz v2.xyz per triangle); mat_idx: n_tris material indices. */
AUR_API AurScene* aur_scene_create(const float*  verts,
                                   int32_t       n_tris,
                                   const int32_t* mat_idx);

AUR_API void aur_scene_destroy(AurScene* scene);

/* Number of BVH nodes (diagnostics). */
AUR_API int32_t aur_scene_node_count(const AurScene* scene);

typedef struct AurTraceParams {
    /* materials: row-major [n_mats][n_bands] */
    const float* absorption;    /* energy absorption coefficient 0..1      */
    const float* scattering;    /* energy scattering coefficient 0..1      */
    const float* transmission;  /* energy transmission coefficient 0..1    */
    int32_t n_mats;
    int32_t n_bands;            /* <= 16                                   */

    float src[3];
    float rcv[3];
    float rcv_radius;           /* receiver detection sphere radius (m)    */

    int32_t n_rays;             /* reverb ray count                        */
    int32_t max_bounces;        /* reverb bounce count                     */
    float   max_time;           /* echogram length (s)                     */
    float   bin_dt;             /* echogram bin width (s)                  */
    int32_t n_bins;

    float speed_of_sound;       /* m/s                                     */
    const float* air_db_per_m;  /* [n_bands] atmospheric attenuation dB/m  */

    int32_t spec_exclude_order; /* pure specular chains of order <= this are
                                   left to the host image sources          */
    float energy_cap;           /* Russian-roulette threshold (rel. energy)*/
    uint64_t seed;
    int32_t n_threads;          /* 0 = hardware concurrency                */

    /* Optional source directivity: energy gain [n_bands][n_dir_samples] vs the angle from
     * src_forward, theta uniform on [0, pi], linear interpolation; 0 samples or NULL = omni. */
    const float* dir_table;
    int32_t n_dir_samples;
    float src_forward[3];
} AurTraceParams;

/* Outputs (caller-allocated, zeroed by the callee):
 *   out_hist : [n_bands][n_bins]  band energy echogram
 *   out_dir  : [n_bins][3]        energy-weighted incoming direction sums
 *   out_stats: [4] = { rays_done, mean_bounces, escaped_rays, total_deposits }
 * Returns 0 on success, 1 bad arguments, 2 n_bands outside 1..16, 3 incomplete AurRbs;
 * CUDA library: 20 no device scene, 100 + cudaError.                       */
AUR_API int32_t aur_trace(const AurScene* scene,
                          const AurTraceParams* p,
                          float* out_hist,
                          float* out_dir,
                          double* out_stats);

/* Reflection-based scattering (Christensen & Rindel 2005): per triangle, the dimensions
 * l >= w (m) of its surface; band_hz[n_bands] the band centres. At each reflection
 * s = 1 - (1 - s_d)(1 - s_mat), s_d from the surface size, the incidence and the path lengths. */
typedef struct AurRbs {
    const float* tri_l;
    const float* tri_w;
    const float* band_hz;
} AurRbs;

AUR_API int32_t aur_trace2(const AurScene* scene,
                           const AurTraceParams* p,
                           const AurRbs* rbs,
                           float* out_hist,
                           float* out_dir,
                           double* out_stats);

/* Product of the per-band transmission of every surface crossed by the segment a->b.
 * out_trans[n_bands] in [0,1]; returns the crossing count.                 */
AUR_API int32_t aur_segment_transmission(const AurScene* scene,
                                         const float a[3], const float b[3],
                                         const float* transmission,
                                         int32_t n_mats, int32_t n_bands,
                                         float* out_trans);

/* Nearest-hit ray cast. Returns the original triangle index or -1;
 * on a hit, *t_out = distance, n_out[3] = unit geometric normal.           */
AUR_API int32_t aur_raycast(const AurScene* scene,
                            const float o[3], const float d[3],
                            float t_max, float* t_out, float n_out[3]);


/* Low-frequency FDTD on a grid compiled in Python: air cells indexed flat,
 * nb[n_air*6] neighbour indices (n_air = ghost cell at zero pressure),
 * per-cell coefficients c_now/c_old/c_nb (Kowalczyk-van Walstijn scheme).
 * Source/receiver taps padded to 8 (ghost index, zero weight); out[n_steps*n_rec].
 * Index arrays are int32 (ABI 2.0): callers refuse an older library. Returns 0 on success. */
AUR_API int32_t aur_fdtd_run(const int32_t* nb,
                             const float* c_now, const float* c_old,
                             const float* c_nb, int64_t n_air,
                             const float* pulse, int32_t n_pulse,
                             const int32_t* src_idx, const float* src_w,
                             int32_t n_src,
                             const int32_t* rec_idx, const float* rec_w,
                             int32_t n_rec,
                             int32_t n_steps, int32_t n_threads,
                             float* out);


/* aur_fdtd_run with filtering walls (f_*, 2 poles per face) and resonant sections (r_*). */
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
                              float* out);

/* During a run, every step writes its index into *counter for a host thread to poll;
 * NULL detaches it. One FDTD run at a time.                                */
AUR_API void aur_fdtd_progress(int64_t* counter);

/* Wave-band grid builder (aur_grid.cpp): the Python reference (fdtd.voxelize / _flood_air,
 * conformal.py) with the same IEEE operations in the same order, threaded. Arrays are
 * C-ordered, cell (i,j,k) at (i*ny + j)*nz + k; tv = n_tri x 9 doubles.                 */
AUR_API int32_t aur_grid_voxelize(const double* tv, int32_t n_tri, const int32_t* tri_mat,
                                  const double* origin, double dx, const int32_t* n,
                                  uint8_t* solid, int32_t* matid, uint8_t* nrmq,
                                  int32_t n_threads);
AUR_API int64_t aur_grid_flood(const uint8_t* solid, const int32_t* n, int64_t seed,
                               uint8_t* air);
AUR_API int32_t aur_grid_fine_fraction(const double* tv, int32_t n_tri, const double* org_f,
                                       double dx_f, const int32_t* n_coarse, int32_t F,
                                       const uint8_t* air_coarse, int32_t n_iter,
                                       double* frac, int32_t n_threads);
AUR_API int64_t aur_grid_wall_faces(const double* tv, int32_t n_tri,
                                    const double* origin_center, double dx, const int32_t* n,
                                    const uint8_t* air, const int64_t* idx, int32_t max_walk,
                                    double* W, double* cell_area, double* stats,
                                    int32_t n_threads);

AUR_API int32_t aur_grid_donate(const int32_t* n, const uint8_t* air, const double* frac,
                                int32_t smooth_sweeps, double t_diff, double* phi,
                                uint8_t* orphan, int32_t n_threads);

/* Integer part of FdtdGrid (aur_grid.cpp), bit for bit: index, neighbours, second
 * neighbours, CPML cell lists and collar.                                               */
AUR_API int64_t aur_grid_structure(const uint8_t* air, const int32_t* n, int32_t* idx,
                                   int32_t* cells, int32_t* nb, double* K, int32_t* nb2,
                                   uint8_t* full, int32_t n_threads);
AUR_API int64_t aur_grid_pml_axis(const int32_t* cells, const int32_t* nb, int64_t n_air,
                                  const int32_t* n, int32_t npml, int32_t axis, int64_t* sel,
                                  int32_t* locm, int32_t n_threads);
AUR_API int64_t aur_grid_wall_list(const int32_t* cells, const int32_t* nb, int64_t n_air,
                                   const int32_t* n, int32_t f, int64_t* wi, uint8_t* inb,
                                   int32_t n_threads);
AUR_API void aur_grid_coefficients(int64_t n_air, double half_lam, double lam2, double lam_dt,
                                   const double* gam, const double* K, const double* s_half,
                                   const double* ph, double* Gamma, double* c_center,
                                   double* c_old, double* c_now, double* c_nb, double* c_drv,
                                   int32_t n_threads);
AUR_API void aur_grid_collar(const int32_t* nb, const int32_t* nb2, int64_t n_air,
                             const uint8_t* pmlm, uint8_t* out, int32_t n_threads);

/* "<library name> <major>.<minor>.<patch>"; callers parse the last word.   */
AUR_API const char* aur_version(void);

#ifdef __cplusplus
}
#endif
#endif /* AUR_NATIVE_H */
