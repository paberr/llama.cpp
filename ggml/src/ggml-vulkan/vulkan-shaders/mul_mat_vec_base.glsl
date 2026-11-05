#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#ifdef MUL_MAT_ID
#define EXPERT_COUNT 8
#endif

#include "mul_mat_vec_iface.glsl"

layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

#ifdef MUL_MAT_ID
    uint nei0;
    uint ne11;
    uint expert_i1;
    uint nbi1;
#else
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
    uint d_row_stride;
    float fuse_scale;
    float fuse_bias;
    uint hc_row_stride; // HC_PRE_FUSION: distance between the hc gate rows of one output element (n_embd)
    uint hc_x_stride;   // HC_PRE_FUSION: column (token) stride of the normed streams (fuse0)
    uint hc_d_stride;   // HC_PRE_FUSION: column (token) stride of the mixed output
#endif
} p;

#ifdef HC_PRE_FUSION
// the rows of a workgroup are the hc gate rows of one output element, hc_row_stride apart
#define MMV_ROW_STEP p.hc_row_stride
#else
#define MMV_ROW_STEP 1u
#endif

#ifndef MUL_MAT_ID
uint mat_vec_d_idx(const uint j, const uint row) {
    return (p.fusion_flags & MAT_VEC_FLAGS_TRANSPOSE_D) != 0 ? row * p.d_row_stride + j : j * p.batch_stride_d + row;
}

#ifdef HC_PRE_FUSION
// DSV4_HC_PRE (gated) fused after the hc gate mat-vec: rows n = 0..NUM_ROWS-1 of this workgroup are the gate of stream n
// of output element first_row. Same arithmetic as dsv4_hc_pre.comp (GATED): fma chain over the streams in order, then scale.
float hc_acc;
void mat_vec_store(const uint j, const uint n, const uint first_row, const uint d_offset, const float g) {
    if (n == 0) {
        hc_acc = 0.0f;
    }
    const float wv = 1.0f / (1.0f + exp(-g));
    hc_acc = fma(float(data_fuse0[j * p.hc_x_stride + first_row + n * p.hc_row_stride]), wv, hc_acc);
    if (n == NUM_ROWS - 1) {
        data_d[d_offset + j * p.hc_d_stride + first_row] = D_TYPE(p.fuse_scale * hc_acc);
    }
}
#endif

// the fused SCALE + UNARY(SILU) after a dense mat-vec, computed exactly like scale.comp and unary.comp (op_silu) on the f32 result
float mat_vec_epilogue(const float r) {
    if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE_SILU) != 0) {
        const float x = float(r) * float(p.fuse_scale) + float(p.fuse_bias);
        return x / (1.0f + exp(-x));
    }
    return r;
}

#ifndef HC_PRE_FUSION
void mat_vec_store(const uint j, const uint n, const uint first_row, const uint d_offset, const float r) {
    data_d[d_offset + mat_vec_d_idx(j, first_row + n)] = D_TYPE(mat_vec_epilogue(r));
}
#endif
#endif

#ifdef MUL_MAT_ID
uint expert_id;
bool expert_skip; // an id of -1 skips the slot, the result row is zeroed
#endif

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
#ifdef MUL_MAT_ID
    const uint expert_i0 = gl_WorkGroupID.y;
#else
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
#endif

#ifndef MUL_MAT_ID
    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }
#else
    const int expert = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
    expert_skip = expert == -1;
    expert_id   = expert_skip ? 0 : uint(expert); // clamp, so that the offsets below stay in range
#endif

    a_offset =
#ifdef MUL_MAT_ID
            expert_id * (p.batch_stride_a / QUANT_K);
#else
            batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    b_offset =
#ifdef MUL_MAT_ID
            (expert_i0 % p.ne11) * p.stride_b + p.expert_i1 * p.batch_stride_b;
#else
            batch_idx * p.batch_stride_b;
#endif
    d_offset =
#ifdef MUL_MAT_ID
            expert_i0 * p.stride_d + p.expert_i1 * p.batch_stride_d;
#else
            batch_idx * p.batch_stride_d;
#endif
}

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                #ifdef MUL_MAT_ID
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
#else
                mat_vec_store(j, n, first_row, d_offset, float(D_TYPE(temp[j][n])));
#endif
            }
        }
    }
}
#else
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];

void reduce_result(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    // subgroupAdd is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup
#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                #ifdef MUL_MAT_ID
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
#else
                mat_vec_store(j, n, first_row, d_offset, float(D_TYPE(temp[j][n])));
#endif
            }
        }
    }
#else
    // sum up partial sums and write back result
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    tmpsh[j][n][0] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                #ifdef MUL_MAT_ID
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(tmpsh[j][n][0]);
#else
                mat_vec_store(j, n, first_row, d_offset, float(D_TYPE(tmpsh[j][n][0])));
#endif
            }
        }
    }
#endif
}
#endif
