/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// MKL VML (Vector Math Library) Helper
//
// Provides vectorized math operations using Intel MKL VML when available,
// with automatic fallback to scalar implementations when MKL is not present.
//
// Usage:
//   sd::vml::erf(n, input, output);  // Vectorized error function
//   sd::vml::exp(n, input, output);  // Vectorized exponential
//   sd::vml::tanh(n, input, output); // Vectorized hyperbolic tangent
//

#ifndef LIBND4J_MKL_VML_HELPER_H
#define LIBND4J_MKL_VML_HELPER_H

#include <system/common.h>
#include <math/templatemath.h>
#include <cmath>

#ifdef HAVE_MKL_VML
// Only include VML headers - do NOT include mkl.h as it conflicts with OpenBLAS cblas.h
// mkl_vml.h provides: vsErf, vdErf, vsExp, vdExp, vsTanh, vdTanh, etc.
#include <mkl_vml.h>
#include <mkl_vml_functions.h>
#endif

namespace sd {
namespace vml {

// ============================================================================
// VML Mode Configuration
// ============================================================================

#ifdef HAVE_MKL_VML
// Set VML accuracy mode - use high accuracy by default
// Options: VML_LA (low accuracy), VML_HA (high accuracy), VML_EP (enhanced performance)
constexpr MKL_INT VML_MODE = VML_HA;

// Minimum array size to use VML (below this, scalar is often faster)
constexpr sd::LongType VML_MIN_SIZE = 32;
#endif

// ============================================================================
// Error Function (erf)
// ============================================================================

/**
 * Vectorized error function for float arrays
 * @param n Number of elements
 * @param input Input array
 * @param output Output array (can be same as input for in-place)
 */
SD_INLINE void erf(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsErf(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    // Scalar fallback
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::erf(input[i]);
    }
}

/**
 * Vectorized error function for double arrays
 */
SD_INLINE void erf(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdErf(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    // Scalar fallback
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::erf(input[i]);
    }
}

// ============================================================================
// Exponential (exp)
// ============================================================================

SD_INLINE void exp(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsExp(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = sd::math::sd_exp<float, float>(input[i]);
    }
}

SD_INLINE void exp(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdExp(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = sd::math::sd_exp<double, double>(input[i]);
    }
}

// ============================================================================
// Hyperbolic Tangent (tanh)
// ============================================================================

SD_INLINE void tanh(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsTanh(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

SD_INLINE void tanh(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdTanh(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

// ============================================================================
// Sigmoid (logistic function): 1 / (1 + exp(-x))
// ============================================================================

SD_INLINE void sigmoid(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        // Compute -x
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < n; ++i) {
            output[i] = -input[i];
        }
        // Compute exp(-x)
        vsExp(static_cast<MKL_INT>(n), output, output);
        // Compute 1 / (1 + exp(-x))
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < n; ++i) {
            output[i] = 1.0f / (1.0f + output[i]);
        }
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = 1.0f / (1.0f + sd::math::sd_exp<float, float>(-input[i]));
    }
}

SD_INLINE void sigmoid(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < n; ++i) {
            output[i] = -input[i];
        }
        vdExp(static_cast<MKL_INT>(n), output, output);
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < n; ++i) {
            output[i] = 1.0 / (1.0 + output[i]);
        }
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = 1.0 / (1.0 + sd::math::sd_exp<double, double>(-input[i]));
    }
}

// ============================================================================
// Square Root (sqrt)
// ============================================================================

SD_INLINE void sqrt(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsSqrt(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = sd::math::sd_sqrt<float, float>(input[i]);
    }
}

SD_INLINE void sqrt(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdSqrt(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = sd::math::sd_sqrt<double, double>(input[i]);
    }
}

// ============================================================================
// Natural Logarithm (log)
// ============================================================================

SD_INLINE void log(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsLn(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = sd::math::sd_log<float, float>(input[i]);
    }
}

SD_INLINE void log(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdLn(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = sd::math::sd_log<double, double>(input[i]);
    }
}

// ============================================================================
// Power (pow)
// ============================================================================

SD_INLINE void pow(sd::LongType n, const float* base, const float* exponent, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsPow(static_cast<MKL_INT>(n), base, exponent, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::pow(base[i], exponent[i]);
    }
}

SD_INLINE void pow(sd::LongType n, const double* base, const double* exponent, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdPow(static_cast<MKL_INT>(n), base, exponent, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::pow(base[i], exponent[i]);
    }
}

// Scalar exponent version
SD_INLINE void powx(sd::LongType n, const float* base, float exponent, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsPowx(static_cast<MKL_INT>(n), base, exponent, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::pow(base[i], exponent);
    }
}

SD_INLINE void powx(sd::LongType n, const double* base, double exponent, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdPowx(static_cast<MKL_INT>(n), base, exponent, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::pow(base[i], exponent);
    }
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

SD_INLINE void sin(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsSin(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::sin(input[i]);
    }
}

SD_INLINE void sin(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdSin(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::sin(input[i]);
    }
}

SD_INLINE void cos(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsCos(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::cos(input[i]);
    }
}

SD_INLINE void cos(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdCos(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::cos(input[i]);
    }
}

// ============================================================================
// Hyperbolic Sine (sinh)
// ============================================================================

SD_INLINE void sinh(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsSinh(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::sinh(input[i]);
    }
}

SD_INLINE void sinh(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdSinh(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::sinh(input[i]);
    }
}

// ============================================================================
// Hyperbolic Cosine (cosh)
// ============================================================================

SD_INLINE void cosh(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsCosh(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::cosh(input[i]);
    }
}

SD_INLINE void cosh(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdCosh(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::cosh(input[i]);
    }
}

// ============================================================================
// Inverse Square Root (1/sqrt(x))
// ============================================================================

SD_INLINE void invsqrt(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsInvSqrt(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = 1.0f / sd::math::sd_sqrt<float, float>(input[i]);
    }
}

SD_INLINE void invsqrt(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdInvSqrt(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = 1.0 / sd::math::sd_sqrt<double, double>(input[i]);
    }
}

// ============================================================================
// Inverse (1/x)
// ============================================================================

SD_INLINE void inv(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsInv(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = 1.0f / input[i];
    }
}

SD_INLINE void inv(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdInv(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = 1.0 / input[i];
    }
}

// ============================================================================
// Absolute Value (abs)
// ============================================================================

SD_INLINE void abs(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsAbs(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::abs(input[i]);
    }
}

SD_INLINE void abs(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdAbs(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = std::abs(input[i]);
    }
}

// ============================================================================
// Square (x^2)
// ============================================================================

SD_INLINE void sqr(sd::LongType n, const float* input, float* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vsSqr(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = input[i] * input[i];
    }
}

SD_INLINE void sqr(sd::LongType n, const double* input, double* output) {
#ifdef HAVE_MKL_VML
    if (n >= VML_MIN_SIZE) {
        vdSqr(static_cast<MKL_INT>(n), input, output);
        return;
    }
#endif
    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < n; ++i) {
        output[i] = input[i] * input[i];
    }
}

// ============================================================================
// Utility: Check if MKL VML is available at runtime
// ============================================================================

SD_INLINE bool isVmlAvailable() {
#ifdef HAVE_MKL_VML
    return true;
#else
    return false;
#endif
}

}  // namespace vml
}  // namespace sd

#endif  // LIBND4J_MKL_VML_HELPER_H
