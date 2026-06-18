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
// Apple Accelerate framework - Utility operations via vDSP
// Provides reverse, sort, ramp, fill, polynomial, and other utility ops
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

//==============================================================================
// REVERSE - Reverse array elements
//==============================================================================

PLATFORM_IMPL(reverse, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get dimensions to reverse along (default: all)
    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    // For 1D array or full reverse
    if (input->rankOf() == 1 || dims.empty()) {
        const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

        if (input->dataType() == DataType::FLOAT32) {
            // Copy to output first
            if (output != input) {
                output->assign(input);
            }
            // vDSP_vrvrs reverses in place
            vDSP_vrvrs(output->bufferAsT<float>(), 1, n);
        } else if (input->dataType() == DataType::DOUBLE) {
            if (output != input) {
                output->assign(input);
            }
            vDSP_vrvrsD(output->bufferAsT<double>(), 1, n);
        }
        return Status::OK;
    }

    // For multi-dimensional, fall back
    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reverse, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    Requirements reqs("REVERSE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    // Only support 1D or full reverse
    reqs.expectTrue(input->rankOf() == 1 || dims.empty(), "Only 1D or full reverse supported");

    return reqs.ok();
}

//==============================================================================
// SORT - Sort array elements (ascending)
//==============================================================================

PLATFORM_IMPL(sort, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    bool descending = block.getBArguments()->size() > 0 ? B_ARG(0) : false;

    // For 1D array
    if (input->rankOf() == 1) {
        const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

        // Copy to output first
        if (output != input) {
            output->assign(input);
        }

        if (input->dataType() == DataType::FLOAT32) {
            if (descending) {
                vDSP_vsort(output->bufferAsT<float>(), n, -1);  // -1 for descending
            } else {
                vDSP_vsort(output->bufferAsT<float>(), n, 1);   // 1 for ascending
            }
        } else if (input->dataType() == DataType::DOUBLE) {
            if (descending) {
                vDSP_vsortD(output->bufferAsT<double>(), n, -1);
            } else {
                vDSP_vsortD(output->bufferAsT<double>(), n, 1);
            }
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(sort, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("SORT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1, "Only 1D sort supported");

    return reqs.ok();
}

//==============================================================================
// LINSPACE - Generate linearly spaced values (like ramp)
//==============================================================================

PLATFORM_IMPL(linspace, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    // Get start, stop from inputs or T args
    double start, stop;
    if (block.width() >= 2) {
        start = INPUT_VARIABLE(0)->e<double>(0);
        stop = INPUT_VARIABLE(1)->e<double>(0);
    } else {
        start = T_ARG(0);
        stop = T_ARG(1);
    }

    const vDSP_Length n = static_cast<vDSP_Length>(output->lengthOf());
    double step = (n > 1) ? (stop - start) / (n - 1) : 0.0;

    if (output->dataType() == DataType::FLOAT32) {
        float startF = static_cast<float>(start);
        float stepF = static_cast<float>(step);
        vDSP_vramp(&startF, &stepF, output->bufferAsT<float>(), 1, n);
    } else if (output->dataType() == DataType::DOUBLE) {
        vDSP_vrampD(&start, &step, output->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(linspace, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("LINSPACE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, nullptr, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(output->dataType()), TYPE_MSG_OUTPUT0);
    reqs.expectTrue(output->rankOf() == 1, "Output must be 1D");

    return reqs.ok();
}

//==============================================================================
// FILL - Fill array with constant value
//==============================================================================

PLATFORM_IMPL(fill, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    double value = T_ARG(0);
    const vDSP_Length n = static_cast<vDSP_Length>(output->lengthOf());

    if (output->dataType() == DataType::FLOAT32) {
        float valF = static_cast<float>(value);
        vDSP_vfill(&valF, output->bufferAsT<float>(), 1, n);
    } else if (output->dataType() == DataType::DOUBLE) {
        vDSP_vfillD(&value, output->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(fill, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("FILL ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, nullptr, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(output->dataType()), TYPE_MSG_OUTPUT0);
    reqs.expectTrue(block.getTArguments()->size() >= 1, "Need fill value T argument");

    return reqs.ok();
}

//==============================================================================
// POLYNOMIAL - Evaluate polynomial: y = c0 + c1*x + c2*x^2 + ...
//==============================================================================

PLATFORM_IMPL(polyval, ENGINE_CPU) {
    auto coeffs = INPUT_VARIABLE(0);  // Polynomial coefficients [c0, c1, c2, ...]
    auto x = INPUT_VARIABLE(1);       // Points to evaluate at
    auto y = OUTPUT_VARIABLE(0);      // Output values

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());
    const vDSP_Length degree = static_cast<vDSP_Length>(coeffs->lengthOf() - 1);

    if (x->dataType() == DataType::FLOAT32) {
        // vDSP_vpoly expects coefficients in reverse order (highest degree first)
        std::vector<float> reversedCoeffs(coeffs->lengthOf());
        const float* cBuf = coeffs->bufferAsT<float>();
        for (vDSP_Length i = 0; i <= degree; i++) {
            reversedCoeffs[i] = cBuf[degree - i];
        }

        vDSP_vpoly(reversedCoeffs.data(), 1,
                   x->bufferAsT<float>(), 1,
                   y->bufferAsT<float>(), 1,
                   n, degree);
    } else if (x->dataType() == DataType::DOUBLE) {
        std::vector<double> reversedCoeffs(coeffs->lengthOf());
        const double* cBuf = coeffs->bufferAsT<double>();
        for (vDSP_Length i = 0; i <= degree; i++) {
            reversedCoeffs[i] = cBuf[degree - i];
        }

        vDSP_vpolyD(reversedCoeffs.data(), 1,
                    x->bufferAsT<double>(), 1,
                    y->bufferAsT<double>(), 1,
                    n, degree);
    }

    return Status::OK;
}

PLATFORM_CHECK(polyval, ENGINE_CPU) {
    auto coeffs = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto y = OUTPUT_VARIABLE(0);

    Requirements reqs("POLYVAL ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, y);
    reqs.expectTrue(accelerateUtils::isContiguous(*coeffs), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT1);
    reqs.expectTrue(coeffs->rankOf() == 1, "Coefficients must be 1D");

    return reqs.ok();
}

//==============================================================================
// INTERP - Linear interpolation
//==============================================================================

PLATFORM_IMPL(interp, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);       // Known x values
    auto y = INPUT_VARIABLE(1);       // Known y values
    auto xNew = INPUT_VARIABLE(2);    // New x values to interpolate at
    auto yNew = OUTPUT_VARIABLE(0);   // Interpolated y values

    // This is a simplified implementation
    // vDSP_vlint does linear interpolation with fractional indices

    const vDSP_Length n = static_cast<vDSP_Length>(xNew->lengthOf());
    const vDSP_Length tableLen = static_cast<vDSP_Length>(y->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();
        const float* xNewBuf = xNew->bufferAsT<float>();
        float* yNewBuf = yNew->bufferAsT<float>();

        float xMin = xBuf[0];
        float xMax = xBuf[tableLen - 1];
        float scale = (tableLen - 1) / (xMax - xMin);

        // Compute fractional indices
        std::vector<float> indices(n);
        for (vDSP_Length i = 0; i < n; i++) {
            indices[i] = (xNewBuf[i] - xMin) * scale;
            // Clamp to valid range
            if (indices[i] < 0) indices[i] = 0;
            if (indices[i] > tableLen - 1) indices[i] = tableLen - 1;
        }

        vDSP_vlint(yBuf, indices.data(), 1, yNewBuf, 1, n, tableLen);
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();
        const double* xNewBuf = xNew->bufferAsT<double>();
        double* yNewBuf = yNew->bufferAsT<double>();

        double xMin = xBuf[0];
        double xMax = xBuf[tableLen - 1];
        double scale = (tableLen - 1) / (xMax - xMin);

        std::vector<double> indices(n);
        for (vDSP_Length i = 0; i < n; i++) {
            indices[i] = (xNewBuf[i] - xMin) * scale;
            if (indices[i] < 0) indices[i] = 0;
            if (indices[i] > tableLen - 1) indices[i] = tableLen - 1;
        }

        vDSP_vlintD(yBuf, indices.data(), 1, yNewBuf, 1, n, tableLen);
    }

    return Status::OK;
}

PLATFORM_CHECK(interp, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto xNew = INPUT_VARIABLE(2);
    auto yNew = OUTPUT_VARIABLE(0);

    Requirements reqs("INTERP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, xNew, yNew);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*xNew), EWS_MSG_INPUT2);
    reqs.expectTrue(accelerateUtils::isContiguous(*yNew), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->rankOf() == 1 && y->rankOf() == 1, "x and y must be 1D");
    reqs.expectTrue(x->lengthOf() == y->lengthOf(), "x and y must have same length");

    return reqs.ok();
}

//==============================================================================
// HAMMING_DISTANCE - Hamming distance between two arrays
//==============================================================================

PLATFORM_IMPL(hamming_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();

        // Compute difference
        std::vector<float> diff(n);
        vDSP_vsub(yBuf, 1, xBuf, 1, diff.data(), 1, n);

        // Compute absolute values
        const int nInt = static_cast<int>(n);
        vvfabsf(diff.data(), diff.data(), &nInt);

        // Count non-zeros (hamming distance)
        float count = 0.0f;
        for (vDSP_Length i = 0; i < n; i++) {
            if (diff[i] != 0.0f) count += 1.0f;
        }
        z->p(0, count);
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();

        std::vector<double> diff(n);
        vDSP_vsubD(yBuf, 1, xBuf, 1, diff.data(), 1, n);

        const int nInt = static_cast<int>(n);
        vvfabs(diff.data(), diff.data(), &nInt);

        double count = 0.0;
        for (vDSP_Length i = 0; i < n; i++) {
            if (diff[i] != 0.0) count += 1.0;
        }
        z->p(0, count);
    }

    return Status::OK;
}

PLATFORM_CHECK(hamming_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("HAMMING_DISTANCE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);
    reqs.expectTrue(z->isScalar(), "Output must be scalar");

    return reqs.ok();
}

//==============================================================================
// COSINE_DISTANCE - Cosine distance: 1 - cos_similarity
//==============================================================================

PLATFORM_IMPL(cosine_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        // Dot product
        float dotProduct = cblas_sdot(n, x->bufferAsT<float>(), 1, y->bufferAsT<float>(), 1);

        // Norms
        float normX = cblas_snrm2(n, x->bufferAsT<float>(), 1);
        float normY = cblas_snrm2(n, y->bufferAsT<float>(), 1);

        // Cosine similarity and distance
        float cosSim = (normX * normY > 0) ? dotProduct / (normX * normY) : 0.0f;
        z->p(0, 1.0f - cosSim);
    } else if (x->dataType() == DataType::DOUBLE) {
        double dotProduct = cblas_ddot(n, x->bufferAsT<double>(), 1, y->bufferAsT<double>(), 1);
        double normX = cblas_dnrm2(n, x->bufferAsT<double>(), 1);
        double normY = cblas_dnrm2(n, y->bufferAsT<double>(), 1);

        double cosSim = (normX * normY > 0) ? dotProduct / (normX * normY) : 0.0;
        z->p(0, 1.0 - cosSim);
    }

    return Status::OK;
}

PLATFORM_CHECK(cosine_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("COSINE_DISTANCE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);
    reqs.expectTrue(z->isScalar(), "Output must be scalar");

    return reqs.ok();
}

//==============================================================================
// EUCLIDEAN_DISTANCE - Euclidean distance (L2)
//==============================================================================

PLATFORM_IMPL(euclidean_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());
    const int nInt = static_cast<int>(n);

    if (x->dataType() == DataType::FLOAT32) {
        // Compute difference
        std::vector<float> diff(n);
        vDSP_vsub(y->bufferAsT<float>(), 1, x->bufferAsT<float>(), 1, diff.data(), 1, n);

        // Compute L2 norm of difference
        float dist = cblas_snrm2(nInt, diff.data(), 1);
        z->p(0, dist);
    } else if (x->dataType() == DataType::DOUBLE) {
        std::vector<double> diff(n);
        vDSP_vsubD(y->bufferAsT<double>(), 1, x->bufferAsT<double>(), 1, diff.data(), 1, n);

        double dist = cblas_dnrm2(nInt, diff.data(), 1);
        z->p(0, dist);
    }

    return Status::OK;
}

PLATFORM_CHECK(euclidean_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("EUCLIDEAN_DISTANCE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);
    reqs.expectTrue(z->isScalar(), "Output must be scalar");

    return reqs.ok();
}

//==============================================================================
// MANHATTAN_DISTANCE - Manhattan distance (L1)
//==============================================================================

PLATFORM_IMPL(manhattan_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());
    const int nInt = static_cast<int>(n);

    if (x->dataType() == DataType::FLOAT32) {
        std::vector<float> diff(n);
        vDSP_vsub(y->bufferAsT<float>(), 1, x->bufferAsT<float>(), 1, diff.data(), 1, n);

        // Compute L1 norm (sum of absolute values)
        float dist = cblas_sasum(nInt, diff.data(), 1);
        z->p(0, dist);
    } else if (x->dataType() == DataType::DOUBLE) {
        std::vector<double> diff(n);
        vDSP_vsubD(y->bufferAsT<double>(), 1, x->bufferAsT<double>(), 1, diff.data(), 1, n);

        double dist = cblas_dasum(nInt, diff.data(), 1);
        z->p(0, dist);
    }

    return Status::OK;
}

PLATFORM_CHECK(manhattan_distance, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("MANHATTAN_DISTANCE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);
    reqs.expectTrue(z->isScalar(), "Output must be scalar");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
