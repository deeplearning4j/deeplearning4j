/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Apple Accelerate framework - FFT operations
// Uses vDSP for optimized Fast Fourier Transform
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <array/NDArrayFactory.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

// Helper to get log2 of a power of 2
static int log2n(int n) {
    int log2 = 0;
    while ((1 << log2) < n) log2++;
    return log2;
}

// Check if n is a power of 2
static bool isPowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

//==============================================================================
// FFT - Complex-to-Complex Fast Fourier Transform
//==============================================================================

PLATFORM_IMPL(fft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Get FFT length from last dimension
    const int n = static_cast<int>(x->sizeAt(-1));
    const int log2N = log2n(n);

    // For batch FFT, compute how many FFTs to perform
    const LongType numBatches = x->lengthOf() / n;

    if (x->dataType() == DataType::FLOAT32) {
        // Create FFT setup
        FFTSetup fftSetup = vDSP_create_fftsetup(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        for (LongType batch = 0; batch < numBatches; batch++) {
            const float* inPtr = x->bufferAsT<float>() + batch * n;
            float* outPtr = z->bufferAsT<float>() + batch * n;

            // Split complex format for vDSP
            // Assuming input is interleaved real/imag pairs
            std::vector<float> realPart(n / 2);
            std::vector<float> imagPart(n / 2);

            // Deinterleave if input is complex interleaved
            for (int i = 0; i < n / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            // Perform FFT (forward = FFT_FORWARD)
            vDSP_fft_zip(fftSetup, &splitComplex, 1, log2N, FFT_FORWARD);

            // Interleave output
            for (int i = 0; i < n / 2; i++) {
                outPtr[2 * i] = realPart[i];
                outPtr[2 * i + 1] = imagPart[i];
            }
        }

        vDSP_destroy_fftsetup(fftSetup);
    } else if (x->dataType() == DataType::DOUBLE) {
        FFTSetupD fftSetup = vDSP_create_fftsetupD(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        for (LongType batch = 0; batch < numBatches; batch++) {
            const double* inPtr = x->bufferAsT<double>() + batch * n;
            double* outPtr = z->bufferAsT<double>() + batch * n;

            std::vector<double> realPart(n / 2);
            std::vector<double> imagPart(n / 2);

            for (int i = 0; i < n / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPDoubleSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2N, FFT_FORWARD);

            for (int i = 0; i < n / 2; i++) {
                outPtr[2 * i] = realPart[i];
                outPtr[2 * i + 1] = imagPart[i];
            }
        }

        vDSP_destroy_fftsetupD(fftSetup);
    }

    return Status::OK;
}

PLATFORM_CHECK(fft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->sizeAt(-1));

    Requirements reqs("FFT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(isPowerOf2(n), "FFT length must be power of 2");

    return reqs.ok();
}

//==============================================================================
// IFFT - Inverse Complex-to-Complex Fast Fourier Transform
//==============================================================================

PLATFORM_IMPL(ifft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->sizeAt(-1));
    const int log2N = log2n(n);
    const LongType numBatches = x->lengthOf() / n;

    if (x->dataType() == DataType::FLOAT32) {
        FFTSetup fftSetup = vDSP_create_fftsetup(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        const float scale = 1.0f / n;

        for (LongType batch = 0; batch < numBatches; batch++) {
            const float* inPtr = x->bufferAsT<float>() + batch * n;
            float* outPtr = z->bufferAsT<float>() + batch * n;

            std::vector<float> realPart(n / 2);
            std::vector<float> imagPart(n / 2);

            for (int i = 0; i < n / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            // Perform IFFT (inverse = FFT_INVERSE)
            vDSP_fft_zip(fftSetup, &splitComplex, 1, log2N, FFT_INVERSE);

            // Scale and interleave output
            for (int i = 0; i < n / 2; i++) {
                outPtr[2 * i] = realPart[i] * scale;
                outPtr[2 * i + 1] = imagPart[i] * scale;
            }
        }

        vDSP_destroy_fftsetup(fftSetup);
    } else if (x->dataType() == DataType::DOUBLE) {
        FFTSetupD fftSetup = vDSP_create_fftsetupD(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        const double scale = 1.0 / n;

        for (LongType batch = 0; batch < numBatches; batch++) {
            const double* inPtr = x->bufferAsT<double>() + batch * n;
            double* outPtr = z->bufferAsT<double>() + batch * n;

            std::vector<double> realPart(n / 2);
            std::vector<double> imagPart(n / 2);

            for (int i = 0; i < n / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPDoubleSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2N, FFT_INVERSE);

            for (int i = 0; i < n / 2; i++) {
                outPtr[2 * i] = realPart[i] * scale;
                outPtr[2 * i + 1] = imagPart[i] * scale;
            }
        }

        vDSP_destroy_fftsetupD(fftSetup);
    }

    return Status::OK;
}

PLATFORM_CHECK(ifft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->sizeAt(-1));

    Requirements reqs("IFFT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(isPowerOf2(n), "FFT length must be power of 2");

    return reqs.ok();
}

//==============================================================================
// RFFT - Real-to-Complex Fast Fourier Transform
//==============================================================================

PLATFORM_IMPL(rfft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->sizeAt(-1));
    const int log2N = log2n(n);
    const LongType numBatches = x->lengthOf() / n;

    if (x->dataType() == DataType::FLOAT32) {
        FFTSetup fftSetup = vDSP_create_fftsetup(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        for (LongType batch = 0; batch < numBatches; batch++) {
            const float* inPtr = x->bufferAsT<float>() + batch * n;
            float* outPtr = z->bufferAsT<float>() + batch * (n + 2);  // rfft output is n/2+1 complex values

            // Pack real data into split complex format
            std::vector<float> realPart(n / 2);
            std::vector<float> imagPart(n / 2);

            // vDSP_ctoz equivalent - split even/odd
            for (int i = 0; i < n / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            // Perform real FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFT_FORWARD);

            // Unpack to standard complex format
            // DC component
            outPtr[0] = splitComplex.realp[0];
            outPtr[1] = 0.0f;

            // Nyquist component (packed in imag[0] by vDSP)
            outPtr[n] = splitComplex.imagp[0];
            outPtr[n + 1] = 0.0f;

            // Other components
            for (int i = 1; i < n / 2; i++) {
                outPtr[2 * i] = splitComplex.realp[i];
                outPtr[2 * i + 1] = splitComplex.imagp[i];
            }
        }

        vDSP_destroy_fftsetup(fftSetup);
    } else if (x->dataType() == DataType::DOUBLE) {
        FFTSetupD fftSetup = vDSP_create_fftsetupD(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        for (LongType batch = 0; batch < numBatches; batch++) {
            const double* inPtr = x->bufferAsT<double>() + batch * n;
            double* outPtr = z->bufferAsT<double>() + batch * (n + 2);

            std::vector<double> realPart(n / 2);
            std::vector<double> imagPart(n / 2);

            for (int i = 0; i < n / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPDoubleSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            vDSP_fft_zripD(fftSetup, &splitComplex, 1, log2N, FFT_FORWARD);

            outPtr[0] = splitComplex.realp[0];
            outPtr[1] = 0.0;
            outPtr[n] = splitComplex.imagp[0];
            outPtr[n + 1] = 0.0;

            for (int i = 1; i < n / 2; i++) {
                outPtr[2 * i] = splitComplex.realp[i];
                outPtr[2 * i + 1] = splitComplex.imagp[i];
            }
        }

        vDSP_destroy_fftsetupD(fftSetup);
    }

    return Status::OK;
}

PLATFORM_CHECK(rfft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->sizeAt(-1));

    Requirements reqs("RFFT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(isPowerOf2(n), "FFT length must be power of 2");

    return reqs.ok();
}

//==============================================================================
// IRFFT - Complex-to-Real Inverse Fast Fourier Transform
//==============================================================================

PLATFORM_IMPL(irfft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Output length is (input_complex_pairs - 1) * 2
    const int nOut = static_cast<int>(z->sizeAt(-1));
    const int log2N = log2n(nOut);
    const LongType numBatches = z->lengthOf() / nOut;

    if (x->dataType() == DataType::FLOAT32) {
        FFTSetup fftSetup = vDSP_create_fftsetup(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        const float scale = 1.0f / (2 * nOut);

        for (LongType batch = 0; batch < numBatches; batch++) {
            const float* inPtr = x->bufferAsT<float>() + batch * (nOut + 2);
            float* outPtr = z->bufferAsT<float>() + batch * nOut;

            std::vector<float> realPart(nOut / 2);
            std::vector<float> imagPart(nOut / 2);

            // Pack into split complex format
            realPart[0] = inPtr[0];
            imagPart[0] = inPtr[nOut];  // Nyquist

            for (int i = 1; i < nOut / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            // Perform inverse real FFT
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFT_INVERSE);

            // Unpack to real output
            for (int i = 0; i < nOut / 2; i++) {
                outPtr[2 * i] = realPart[i] * scale;
                outPtr[2 * i + 1] = imagPart[i] * scale;
            }
        }

        vDSP_destroy_fftsetup(fftSetup);
    } else if (x->dataType() == DataType::DOUBLE) {
        FFTSetupD fftSetup = vDSP_create_fftsetupD(log2N, FFT_RADIX2);
        if (fftSetup == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        const double scale = 1.0 / (2 * nOut);

        for (LongType batch = 0; batch < numBatches; batch++) {
            const double* inPtr = x->bufferAsT<double>() + batch * (nOut + 2);
            double* outPtr = z->bufferAsT<double>() + batch * nOut;

            std::vector<double> realPart(nOut / 2);
            std::vector<double> imagPart(nOut / 2);

            realPart[0] = inPtr[0];
            imagPart[0] = inPtr[nOut];

            for (int i = 1; i < nOut / 2; i++) {
                realPart[i] = inPtr[2 * i];
                imagPart[i] = inPtr[2 * i + 1];
            }

            DSPDoubleSplitComplex splitComplex;
            splitComplex.realp = realPart.data();
            splitComplex.imagp = imagPart.data();

            vDSP_fft_zripD(fftSetup, &splitComplex, 1, log2N, FFT_INVERSE);

            for (int i = 0; i < nOut / 2; i++) {
                outPtr[2 * i] = realPart[i] * scale;
                outPtr[2 * i + 1] = imagPart[i] * scale;
            }
        }

        vDSP_destroy_fftsetupD(fftSetup);
    }

    return Status::OK;
}

PLATFORM_CHECK(irfft, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int nOut = static_cast<int>(z->sizeAt(-1));

    Requirements reqs("IRFFT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(isPowerOf2(nOut), "Output length must be power of 2");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
