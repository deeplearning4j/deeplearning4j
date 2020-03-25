/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include <ops/declarable/helpers/transforms.h>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void clipByNorm_(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {

    const int rank = input.rankOf();
    const auto norm2 = input.reduceAlongDimension(reduce::Norm2, dimensions);

    const T normActual = norm2.e<T>(0);
    const T normClip   = clipNorm.e<T>(0);

    if (isInplace) {

        if(norm2.lengthOf() == 1) {

            if(normActual > normClip)
                input *= (normClip / normActual);
        }
        else {

            auto listOfInSubArrs = input.allTensorsAlongDimension(dimensions);

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i++) {
                    const T iNormActual = norm2.e<T>(i);
                    if (iNormActual > normClip)
                        *listOfInSubArrs.at(i) *= normClip / iNormActual;
                }
            };
            samediff::Threads::parallel_tad(func, 0, listOfInSubArrs.size());
        }
    }
    else {

        if(norm2.lengthOf() == 1) {

            if(normActual > normClip)
                output.assign(input * (normClip / normActual));
            else
                output.assign(input);
        }
        else {

            auto listOfInSubArrs  = input.allTensorsAlongDimension(dimensions);
            auto listOfOutSubArrs = output.allTensorsAlongDimension(dimensions);

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i++) {
                    auto inputSubArr = listOfInSubArrs.at(i);
                    auto outputSubArr = listOfOutSubArrs.at(i);
                    outputSubArr->assign(inputSubArr);

                    const T iNormActual = norm2.e<T>(i);

                    if (iNormActual > clipNorm.e<T>(0))
                        *outputSubArr *= clipNorm / iNormActual;
                }
            };
            samediff::Threads::parallel_tad(func, 0, listOfInSubArrs.size());
        }
    }
}

//////////////////////////////////////////////////////////////////////////
void clipByNorm(sd::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
    BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
}


    template <typename T>
    static void clipByGlobalNorm_(std::vector<NDArray*> const& inputs, double clipNorm, sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        T globalNorm = 0; //NDArrayFactory::create<T>(0, inputs[0]->getContext()); //sqrt(sum([l2norm(t)**2 for t in t_list]))
//        PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(sumT : globalNorm)
        for (size_t i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto l2norm = input->reduceNumber(reduce::Norm2);
            globalNorm += l2norm.t<T>(0) * l2norm.t<T>(0);
        }

        //globalNorm.applyTransform(transform::Sqrt, nullptr, nullptr);// = sd::math::nd4j_sqrt(globalNorm);
        auto normS = sd::math::nd4j_sqrt<T,T>(globalNorm);
        outputs[inputs.size()]->p(0, normS);

        const T factor = clipNorm / normS;

//        PRAGMA_OMP_PARALLEL_FOR
        for (size_t e = 0; e < inputs.size(); e++) {
            // all-reduce
            auto input = inputs[e];
            auto output = outputs[e];

            if (normS <= clipNorm) {
                output->assign(input);
            }
            else {

                auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
                input->applyLambda<T>(lambda, *output);
            }
        }
    }
    void clipByGlobalNorm(sd::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (std::vector<NDArray*> const& inputs, double clipNorm, sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void clipByNormBP_(const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {

    const int rank = input.rankOf();

    auto norm2 = input.reduceAlongDimension(reduce::Norm2, dimensions);

    if(norm2.lengthOf() == 1) {

        const T N = norm2.e<T>(0);

        auto cn = clipNorm.e<T>(0);

        if(N > cn) {

            const T sumOfProd = (input * gradO).reduceNumber(reduce::Sum).e<T>(0);    // reduce to scalar
            const T factor1 = static_cast<T>(1.f) / N;
            const T factor3 = factor1 / (N * N);                                            // 1 / (N*N*N)

            auto lambda = LAMBDA_TT(elem1, elem2, cn, sumOfProd, factor1, factor3) {
                return cn * (factor1 * elem2 - factor3 * elem1 * sumOfProd);
            };

            (const_cast<NDArray&>(input)).applyPairwiseLambda<T>(const_cast<NDArray&>(gradO), lambda, gradI);
        }
        else
            gradI.assign(gradO);
    }
    else {

        auto gradISubArrs = gradI.allTensorsAlongDimension({dimensions});
        auto gradOSubArrs = gradO.allTensorsAlongDimension({dimensions});
        auto inputSubArrs = input.allTensorsAlongDimension({dimensions});

        auto cn = clipNorm.e<T>(0);

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                T N = norm2.e<T>(i);

                auto gradOSubArr = gradOSubArrs.at(i);
                auto gradISubArr = gradISubArrs.at(i);

                if (N > cn) {
                    auto inputSubArr = inputSubArrs.at(i);
                    const T sumOfProd = (*inputSubArr * *gradOSubArr).reduceNumber(reduce::Sum).e<T>(0);    // reduce to scalar
                    const T factor1 = static_cast<T>(1.f) / N;
                    const T factor3 = factor1 / (N * N);                                            // 1 / (N*N*N)

                    auto lambda = LAMBDA_TT(elem1, elem2, cn, sumOfProd, factor1, factor3) {
                        return cn * (factor1 * elem2 - factor3 * elem1 * sumOfProd);
                    };

                    inputSubArr->applyPairwiseLambda<T>(*gradOSubArr, lambda, *gradISubArr);
                } else
                    gradISubArr->assign(gradOSubArr);
            }
        };
        samediff::Threads::parallel_tad(func, 0, gradISubArrs.size());
    }
}

    void clipByNormBP(sd::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
        BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBP_, (input, gradO, gradI, dimensions, clipNorm), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByNormBP_, (const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm), FLOAT_TYPES);


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void clipByAveraged_(NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {

    auto cn = clipNorm.e<T>(0);
    if (dimensions.size() == 0) {
        // all-reduce
        T n2 = input.reduceNumber(reduce::Norm2).e<T>(0) / input.lengthOf();
        if (n2 <= cn) {
            if (!isInplace)
                output.assign(input);
        }
        else {
            const T factor = cn / n2;
            auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
            input.applyLambda<T>(lambda, output);
        }
    }
    else {
        // along dimension
        auto norm2 = input.reduceAlongDimension(reduce::Norm2, dimensions, false);
        if (!isInplace)
                output.assign(input);
        auto tads = output.allTensorsAlongDimension(dimensions);
        // TODO: make this CUDA-compliant somehow
        for (int e = 0; e < tads.size(); e++) {
            T n2 = norm2.e<T>(e) / tads.at(e)->lengthOf();
            const T factor = cn / n2;
            if (n2 > cn) {
                auto lambda = LAMBDA_T(_x, factor) {return _x * factor;};
                tads.at(e)->applyLambda<T>(lambda, output);
            }
        }
    }
}

    void clipByAveraged(sd::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByAveraged_, (input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByAveraged_, (NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace), FLOAT_TYPES);

/*
    if (d1 > params[1])
    return params[1];
    else if (d1 < params[0])
    return params[0];
    else return d1;
*/

    template <typename T>
    static void clipByValue_(NDArray& input, double leftBound, double rightBound, NDArray& output) {
        auto routine = LAMBDA_T(_x, leftBound, rightBound) {
            if (_x > rightBound) return rightBound;
            if (_x < leftBound)  return leftBound;
            return _x;
        };

        input.applyLambda<T>(routine, output);
    }

    void clipByValue(sd::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

}
}
}
