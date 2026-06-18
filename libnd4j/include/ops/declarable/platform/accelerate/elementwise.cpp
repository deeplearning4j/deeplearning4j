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
// Apple Accelerate framework - Element-wise operations via vDSP
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_ACCELERATE

//////////////////////////////////////////////////////////////////////////
// ReLU activation using vDSP
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const LongType length = input->lengthOf();

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();
        const float zero = 0.0f;

        // vDSP_vthr: vector threshold - clips values below threshold to threshold
        // For ReLU: max(0, x), we use threshold at 0
        vDSP_vthr(inBuf, 1, &zero, outBuf, 1, static_cast<vDSP_Length>(length));
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();
        const double zero = 0.0;

        vDSP_vthrD(inBuf, 1, &zero, outBuf, 1, static_cast<vDSP_Length>(length));
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE RELU OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Tanh activation using vDSP/vForce
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(tanh, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const int length = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // vvtanhf: vectorized tanh for float
        vvtanhf(outBuf, inBuf, &length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        // vvtanh: vectorized tanh for double
        vvtanh(outBuf, inBuf, &length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(tanh, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE TANH OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Sigmoid activation using vDSP/vForce
// sigmoid(x) = 1 / (1 + exp(-x))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const int length = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Step 1: negate input -> -x
        float negOne = -1.0f;
        vDSP_vsmul(inBuf, 1, &negOne, outBuf, 1, static_cast<vDSP_Length>(length));

        // Step 2: exp(-x)
        vvexpf(outBuf, outBuf, &length);

        // Step 3: 1 + exp(-x)
        float one = 1.0f;
        vDSP_vsadd(outBuf, 1, &one, outBuf, 1, static_cast<vDSP_Length>(length));

        // Step 4: 1 / (1 + exp(-x))
        vvrecf(outBuf, outBuf, &length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        // Step 1: negate input -> -x
        double negOne = -1.0;
        vDSP_vsmulD(inBuf, 1, &negOne, outBuf, 1, static_cast<vDSP_Length>(length));

        // Step 2: exp(-x)
        vvexp(outBuf, outBuf, &length);

        // Step 3: 1 + exp(-x)
        double one = 1.0;
        vDSP_vsaddD(outBuf, 1, &one, outBuf, 1, static_cast<vDSP_Length>(length));

        // Step 4: 1 / (1 + exp(-x))
        vvrec(outBuf, outBuf, &length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE SIGMOID OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softmax using vDSP
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softmax, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get the dimension for softmax (default is last dimension)
    int dimension = input->rankOf() - 1;
    if (block.getIArguments()->size() > 0) {
        dimension = INT_ARG(0);
        if (dimension < 0) {
            dimension += input->rankOf();
        }
    }

    // For 1D or when softmax is along the entire flattened array
    if (input->rankOf() == 1 || (input->rankOf() == 2 && dimension == 1 && input->sizeAt(0) == 1)) {
        const int length = static_cast<int>(input->lengthOf());

        if (input->dataType() == DataType::FLOAT32) {
            const float* inBuf = input->bufferAsT<float>();
            float* outBuf = output->bufferAsT<float>();

            // Step 1: Find max value for numerical stability
            float maxVal;
            vDSP_maxv(inBuf, 1, &maxVal, static_cast<vDSP_Length>(length));

            // Step 2: Subtract max from all elements (x - max)
            float negMax = -maxVal;
            vDSP_vsadd(inBuf, 1, &negMax, outBuf, 1, static_cast<vDSP_Length>(length));

            // Step 3: exp(x - max)
            vvexpf(outBuf, outBuf, &length);

            // Step 4: Sum of exp values
            float sum;
            vDSP_sve(outBuf, 1, &sum, static_cast<vDSP_Length>(length));

            // Step 5: Divide by sum
            vDSP_vsdiv(outBuf, 1, &sum, outBuf, 1, static_cast<vDSP_Length>(length));
        } else if (input->dataType() == DataType::DOUBLE) {
            const double* inBuf = input->bufferAsT<double>();
            double* outBuf = output->bufferAsT<double>();

            // Step 1: Find max value
            double maxVal;
            vDSP_maxvD(inBuf, 1, &maxVal, static_cast<vDSP_Length>(length));

            // Step 2: Subtract max
            double negMax = -maxVal;
            vDSP_vsaddD(inBuf, 1, &negMax, outBuf, 1, static_cast<vDSP_Length>(length));

            // Step 3: exp(x - max)
            vvexp(outBuf, outBuf, &length);

            // Step 4: Sum
            double sum;
            vDSP_sveD(outBuf, 1, &sum, static_cast<vDSP_Length>(length));

            // Step 5: Divide by sum
            vDSP_vsdivD(outBuf, 1, &sum, outBuf, 1, static_cast<vDSP_Length>(length));
        }
    } else {
        // For multi-dimensional softmax, fall back to row-by-row processing
        // This handles 2D case where softmax is applied along rows
        if (input->rankOf() == 2 && dimension == 1) {
            const LongType numRows = input->sizeAt(0);
            const int numCols = static_cast<int>(input->sizeAt(1));

            if (input->dataType() == DataType::FLOAT32) {
                for (LongType row = 0; row < numRows; row++) {
                    const float* inBuf = input->bufferAsT<float>() + row * numCols;
                    float* outBuf = output->bufferAsT<float>() + row * numCols;

                    float maxVal;
                    vDSP_maxv(inBuf, 1, &maxVal, static_cast<vDSP_Length>(numCols));

                    float negMax = -maxVal;
                    vDSP_vsadd(inBuf, 1, &negMax, outBuf, 1, static_cast<vDSP_Length>(numCols));

                    vvexpf(outBuf, outBuf, &numCols);

                    float sum;
                    vDSP_sve(outBuf, 1, &sum, static_cast<vDSP_Length>(numCols));

                    vDSP_vsdiv(outBuf, 1, &sum, outBuf, 1, static_cast<vDSP_Length>(numCols));
                }
            } else if (input->dataType() == DataType::DOUBLE) {
                for (LongType row = 0; row < numRows; row++) {
                    const double* inBuf = input->bufferAsT<double>() + row * numCols;
                    double* outBuf = output->bufferAsT<double>() + row * numCols;

                    double maxVal;
                    vDSP_maxvD(inBuf, 1, &maxVal, static_cast<vDSP_Length>(numCols));

                    double negMax = -maxVal;
                    vDSP_vsaddD(inBuf, 1, &negMax, outBuf, 1, static_cast<vDSP_Length>(numCols));

                    vvexp(outBuf, outBuf, &numCols);

                    double sum;
                    vDSP_sveD(outBuf, 1, &sum, static_cast<vDSP_Length>(numCols));

                    vDSP_vsdivD(outBuf, 1, &sum, outBuf, 1, static_cast<vDSP_Length>(numCols));
                }
            }
        } else {
            // For other cases, we need to fall through to the generic implementation
            return sd::Status::KERNEL_FAILURE;
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(softmax, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get the dimension for softmax
    int dimension = input->rankOf() - 1;
    if (block.getIArguments()->size() > 0) {
        dimension = INT_ARG(0);
        if (dimension < 0) {
            dimension += input->rankOf();
        }
    }

    Requirements req("ACCELERATE SOFTMAX OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    // Only support 1D arrays or 2D arrays with softmax along last dimension
    req.expectTrue(input->rankOf() <= 2, "Only 1D and 2D arrays are currently supported");
    if (input->rankOf() == 2) {
        req.expectTrue(dimension == 1, "For 2D arrays, only softmax along last dimension (dim=1) is supported");
    }

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ReLU6 activation: min(max(0, x), 6)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(relu6, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();
        const float zero = 0.0f;
        const float six = 6.0f;

        // Step 1: max(0, x) using vDSP_vthr
        vDSP_vthr(inBuf, 1, &zero, outBuf, 1, length);

        // Step 2: min(result, 6) using vDSP_vclip
        vDSP_vclip(outBuf, 1, &zero, &six, outBuf, 1, length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();
        const double zero = 0.0;
        const double six = 6.0;

        vDSP_vthrD(inBuf, 1, &zero, outBuf, 1, length);
        vDSP_vclipD(outBuf, 1, &zero, &six, outBuf, 1, length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(relu6, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE RELU6 OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Leaky ReLU: max(alpha * x, x) where alpha is typically 0.01
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(lrelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get alpha from T arguments (default 0.01)
    float alpha = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 0.01f;

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());
    const int n = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Allocate temp buffer for alpha * x
        std::vector<float> temp(n);

        // Compute alpha * x
        vDSP_vsmul(inBuf, 1, &alpha, temp.data(), 1, length);

        // Compute max(alpha * x, x)
        vDSP_vmax(temp.data(), 1, inBuf, 1, outBuf, 1, length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();
        double alphaD = static_cast<double>(alpha);

        std::vector<double> temp(n);

        vDSP_vsmulD(inBuf, 1, &alphaD, temp.data(), 1, length);
        vDSP_vmaxD(temp.data(), 1, inBuf, 1, outBuf, 1, length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(lrelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE LRELU OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ELU: x if x > 0, else alpha * (exp(x) - 1)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(elu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get alpha from T arguments (default 1.0)
    float alpha = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f;

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());
    const int n = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Allocate temp buffers
        std::vector<float> expResult(n);
        std::vector<float> negPart(n);

        // Compute exp(x)
        vvexpf(expResult.data(), inBuf, &n);

        // Compute exp(x) - 1
        float negOne = -1.0f;
        vDSP_vsadd(expResult.data(), 1, &negOne, expResult.data(), 1, length);

        // Compute alpha * (exp(x) - 1)
        vDSP_vsmul(expResult.data(), 1, &alpha, negPart.data(), 1, length);

        // For each element: if x > 0, use x; else use alpha * (exp(x) - 1)
        // Use vDSP_vthr to get max(0, x) for positive part
        const float zero = 0.0f;
        std::vector<float> posPart(n);
        vDSP_vthr(inBuf, 1, &zero, posPart.data(), 1, length);

        // Create mask for negative values: where x <= 0
        // We use: result = posPart + negPart * mask
        // where mask is 1 where x <= 0, 0 otherwise

        for (int i = 0; i < n; i++) {
            outBuf[i] = inBuf[i] > 0 ? inBuf[i] : negPart[i];
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();
        double alphaD = static_cast<double>(alpha);

        std::vector<double> expResult(n);
        std::vector<double> negPart(n);

        vvexp(expResult.data(), inBuf, &n);

        double negOne = -1.0;
        vDSP_vsaddD(expResult.data(), 1, &negOne, expResult.data(), 1, length);

        vDSP_vsmulD(expResult.data(), 1, &alphaD, negPart.data(), 1, length);

        for (int i = 0; i < n; i++) {
            outBuf[i] = inBuf[i] > 0 ? inBuf[i] : negPart[i];
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE ELU OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SELU - Scaled ELU: lambda * (x if x > 0 else alpha * (exp(x) - 1))
// lambda = 1.0507, alpha = 1.67326
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(selu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const float lambda = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());
    const int n = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        std::vector<float> expResult(n);
        std::vector<float> negPart(n);

        // Compute exp(x)
        vvexpf(expResult.data(), inBuf, &n);

        // Compute exp(x) - 1
        float negOne = -1.0f;
        vDSP_vsadd(expResult.data(), 1, &negOne, expResult.data(), 1, length);

        // Compute alpha * (exp(x) - 1)
        float alphaF = alpha;
        vDSP_vsmul(expResult.data(), 1, &alphaF, negPart.data(), 1, length);

        // Select and scale by lambda
        for (int i = 0; i < n; i++) {
            outBuf[i] = lambda * (inBuf[i] > 0 ? inBuf[i] : negPart[i]);
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();
        double lambdaD = static_cast<double>(lambda);
        double alphaD = static_cast<double>(alpha);

        std::vector<double> expResult(n);
        std::vector<double> negPart(n);

        vvexp(expResult.data(), inBuf, &n);

        double negOne = -1.0;
        vDSP_vsaddD(expResult.data(), 1, &negOne, expResult.data(), 1, length);

        vDSP_vsmulD(expResult.data(), 1, &alphaD, negPart.data(), 1, length);

        for (int i = 0; i < n; i++) {
            outBuf[i] = lambdaD * (inBuf[i] > 0 ? inBuf[i] : negPart[i]);
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(selu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE SELU OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Swish: x * sigmoid(x) = x / (1 + exp(-x))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(swish, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());
    const int n = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Step 1: -x
        float negOne = -1.0f;
        vDSP_vsmul(inBuf, 1, &negOne, outBuf, 1, length);

        // Step 2: exp(-x)
        vvexpf(outBuf, outBuf, &n);

        // Step 3: 1 + exp(-x)
        float one = 1.0f;
        vDSP_vsadd(outBuf, 1, &one, outBuf, 1, length);

        // Step 4: x / (1 + exp(-x))
        vDSP_vdiv(outBuf, 1, inBuf, 1, outBuf, 1, length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        double negOne = -1.0;
        vDSP_vsmulD(inBuf, 1, &negOne, outBuf, 1, length);

        vvexp(outBuf, outBuf, &n);

        double one = 1.0;
        vDSP_vsaddD(outBuf, 1, &one, outBuf, 1, length);

        vDSP_vdivD(outBuf, 1, inBuf, 1, outBuf, 1, length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(swish, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE SWISH OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softplus: log(1 + exp(x))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softplus, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());
    const int n = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Step 1: exp(x)
        vvexpf(outBuf, inBuf, &n);

        // Step 2: 1 + exp(x)
        float one = 1.0f;
        vDSP_vsadd(outBuf, 1, &one, outBuf, 1, length);

        // Step 3: log(1 + exp(x))
        vvlogf(outBuf, outBuf, &n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        vvexp(outBuf, inBuf, &n);

        double one = 1.0;
        vDSP_vsaddD(outBuf, 1, &one, outBuf, 1, length);

        vvlog(outBuf, outBuf, &n);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(softplus, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE SOFTPLUS OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Hard Sigmoid: max(0, min(1, (x + 3) / 6))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(hardsigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Step 1: x + 3
        float three = 3.0f;
        vDSP_vsadd(inBuf, 1, &three, outBuf, 1, length);

        // Step 2: (x + 3) / 6
        float six = 6.0f;
        vDSP_vsdiv(outBuf, 1, &six, outBuf, 1, length);

        // Step 3: clip to [0, 1]
        float zero = 0.0f;
        float one = 1.0f;
        vDSP_vclip(outBuf, 1, &zero, &one, outBuf, 1, length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        double three = 3.0;
        vDSP_vsaddD(inBuf, 1, &three, outBuf, 1, length);

        double six = 6.0;
        vDSP_vsdivD(outBuf, 1, &six, outBuf, 1, length);

        double zero = 0.0;
        double one = 1.0;
        vDSP_vclipD(outBuf, 1, &zero, &one, outBuf, 1, length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(hardsigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE HARDSIGMOID OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softsign: x / (1 + |x|)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softsign, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const vDSP_Length length = static_cast<vDSP_Length>(input->lengthOf());
    const int n = static_cast<int>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Step 1: |x|
        std::vector<float> absX(n);
        vvfabsf(absX.data(), inBuf, &n);

        // Step 2: 1 + |x|
        float one = 1.0f;
        vDSP_vsadd(absX.data(), 1, &one, absX.data(), 1, length);

        // Step 3: x / (1 + |x|)
        vDSP_vdiv(absX.data(), 1, inBuf, 1, outBuf, 1, length);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        std::vector<double> absX(n);
        vvfabs(absX.data(), inBuf, &n);

        double one = 1.0;
        vDSP_vsaddD(absX.data(), 1, &one, absX.data(), 1, length);

        vDSP_vdivD(absX.data(), 1, inBuf, 1, outBuf, 1, length);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(softsign, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("ACCELERATE SOFTSIGN OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*output), "Output must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
