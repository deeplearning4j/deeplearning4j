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
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <NDArrayFactory.h>
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>
#include <Loops.h>
#include <graph/RandomGenerator.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void triuBP_(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    auto dOdI = NDArray(&gradO);                // dO/dI
    const_cast<NDArray&>(input).fillAsTriangular<T>(0, diagonal, dOdI.sizeAt(-1), dOdI, 'b');
    int dLen = dOdI.lengthOf();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i += increment) {
            if (dOdI.t<T>(i) != static_cast<T>(0.f))
                dOdI.t<T>(i) = static_cast<T>(1.f);
        }
    };
    samediff::Threads::parallel_for(func, 0, dLen);

    // FIXME: !!!
    gradI.assign(dOdI * gradO);                          // chain rule: dLoss/dI = dO/dI * dLoss/dO
}

    void triuBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {
        BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (context, input, gradO, gradI, diagonal), LIBND4J_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void trace_(const NDArray& input, NDArray& output) {
    const int inRank = input.rankOf();
    auto setOfSubArrs = input.allTensorsAlongDimension({inRank-2, inRank-1});

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i += increment)
            output.p(i, setOfSubArrs.at(i)->getTrace());
    };
    samediff::Threads::parallel_for(func, 0, setOfSubArrs.size());
}

    void trace(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), trace_, (input, output), LIBND4J_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template <typename T>
void randomShuffle_(NDArray& input, NDArray& output, nd4j::graph::RandomGenerator& rng, const bool isInplace) {

    // check edge cases first
    int temp;
    const int firstDim = input.sizeAt(0);
    if(input.lengthOf() == 1 || firstDim == 1) {

        if(!isInplace)
            output.assign(input);
    }
    else if (input.isVector() || shape::isLikeVector(input.getShapeInfo(), temp)) {

        // apply Fisher-Yates shuffle
        if(isInplace) {
            //PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance()->tadThreshold())
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;
                if(i == r)
                    continue;
                T t0 = input.t<T>(i);
                T t1 = input.t<T>(r);
                //math::nd4j_swap<T>(input(i), input(r));
                input.t<T>(i) = t1;
                input.t<T>(r) = t0;
            }
        }
        else {
            std::vector<int> indices(firstDim);
            std::iota(indices.begin(), indices.end(), 0);
            output.p<T>(Nd4jLong(0), input.e<T>(0));

            // FIXME: parallelism!!
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;
                output.t<T>(i) = input.t<T>(indices[r]);
                if(i == r)
                    continue;

                output.t<T>(r) = input.t<T>(indices[i]);
                math::nd4j_swap<int>(indices[i], indices[r]);
            }
            rng.rewindH(firstDim-1);
        }
    }
    else {

        // evaluate sub-arrays list of input array through all dimensions excluding first one
        std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input.rankOf(), {0});
        auto subArrsListIn = input.allTensorsAlongDimension(dimensions);

        // apply Fisher-Yates shuffle
        if(isInplace) {
            //PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance()->elementwiseThreshold())
            for(int i = firstDim - 1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;

                if(i == r)
                    continue;
                subArrsListIn.at(i)->swapUnsafe(*subArrsListIn.at(r));
            }
        }
        else {
            // evaluate sub-arrays list of output array through all dimensions excluding first one
            auto subArrsListOut = output.allTensorsAlongDimension(dimensions);
            std::vector<int> indices(firstDim);
            std::iota(indices.begin(), indices.end(), 0);
            bool isZeroShuffled = false;
            //PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance()->tadThreshold())
            for(int i = firstDim - 1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;
                subArrsListOut.at(i)->assign(subArrsListIn.at(indices[r]));
                if(r == 0)
                    isZeroShuffled = true;
                if(i == r)
                    continue;
                subArrsListOut.at(r)->assign(subArrsListIn.at(indices[i]));
                math::nd4j_swap<int>(indices[i], indices[r]);
            }
            if(!isZeroShuffled)
                subArrsListOut.at(0)->assign(subArrsListIn.at(0));
        }
        rng.rewindH(firstDim-1);
    }

}

    void randomShuffle(nd4j::LaunchContext * context, NDArray& input, NDArray& output, nd4j::graph::RandomGenerator& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (input, output, rng, isInplace), LIBND4J_TYPES);
    }


//////////////////////////////////////////////////////////////////////////
template<typename T>
void pad_(const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, const NDArray& padValue) {

    const T* x = input.bufferAsT<T>();
          T* z = output.bufferAsT<T>();

    const Nd4jLong* xShape  = input.shapeOf();
    const Nd4jLong* zShape  = output.shapeOf();

    const int rank = input.rankOf();  // both input and output have the same rank
    const int rankMinusOne = rank - 1;

    const auto zLen = output.lengthOf();

    if(mode == 0) { // CONSTANT case

        const T padVal = padValue.e<T>(0);

        auto func = PRAGMA_THREADS_FOR {
            Nd4jLong coords[MAX_RANK];
            for (auto i = start; i < stop; i += increment) {
                shape::index2coords(i, output.getShapeInfo(), coords);
                const auto zOffset = shape::getOffset(output.getShapeInfo(), coords);

                bool within = true;
                for (int j = rankMinusOne; j >= 0; --j) {
                    if (xShape[j] == zShape[j]) continue;
                    const auto left = paddings.e<Nd4jLong>(j, 0);
                    if (coords[j] < left || coords[j] >= left + xShape[j]) {
                        within = false;
                        break;
                    }
                    else { coords[j] = coords[j] - left; }
                }

                if (within)
                    z[zOffset] = x[shape::getOffset(input.getShapeInfo(), coords)];
                else
                    z[zOffset] = padVal;
            }
        };

        samediff::Threads::parallel_tad(func, 0, zLen);
    }
    else {  // REFLECT and SYMMETRIC cases

        const Nd4jLong shift1 = mode == 1 ? 0 : 1;         // REFLECT : SYMMETRIC
        const Nd4jLong shift2 = mode == 1 ? 2 : 1;         // REFLECT : SYMMETRIC

        auto func = PRAGMA_THREADS_FOR {
            Nd4jLong coords[MAX_RANK];
            for (auto i = start; i < stop; i += increment) {
                shape::index2coords(i, output.getShapeInfo(), coords);
                const auto zOffset = shape::getOffset(output.getShapeInfo(), coords);

                for (int j = rankMinusOne; j >= 0; --j) {

                    if (xShape[j] == zShape[j]) continue;
                    coords[j] = coords[j] - paddings.e<Nd4jLong>(j, 0);                             // are ready to fill middle (within input dimension range)
                    if (coords[j] < 0) coords[j] = -coords[j] - shift1;                // means fill from left
                    else if (coords[j] >= xShape[j]) coords[j] = 2 * xShape[j] - coords[j] - shift2; // means fill from right
                }

                const auto xOffset = shape::getOffset(input.getShapeInfo(), coords);
                z[zOffset] = x[xOffset];
            }
        };

        samediff::Threads::parallel_tad(func, 0, zLen);
    }
}

// //////////////////////////////////////////////////////////////////////////
// template<typename T>
// void pad2_(const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {

//     const int rank = output.rankOf();
//     std::vector<int> dimsToExclude(rank);
//     std::iota(dimsToExclude.begin(), dimsToExclude.end(), 0);             // fill with 0, 1, ... rank-1

//     Nd4jLong numLeft    = paddings.e<Nd4jLong>(rank-1,0);
//     Nd4jLong numRight   = paddings.e<Nd4jLong>(rank-1,1);
//     Nd4jLong inDimSize  = input.sizeAt(rank-1);
//     Nd4jLong outDimSize = output.sizeAt(rank-1);

//     std::vector<std::vector<Nd4jLong>> outIdx = { std::vector<Nd4jLong>(2*rank), {numLeft, numLeft + inDimSize}, {0, numLeft}, {numLeft + inDimSize, outDimSize} };

//     for(int i = 0; i < rank-1; ++i) {
//         outIdx[0][2*i]     = paddings.e<Nd4jLong>(i, 0);
//         outIdx[0][2*i + 1] = outIdx[0][2*i] + input.sizeAt(i);
//     }
//     outIdx[0][2*rank-1] = outIdx[0][2*rank-2] = 0;

//     // ***** populate innermost sub-arrays firstly ***** //
//     dimsToExclude.pop_back();

//     Nd4jLong startL = mode == 1 ? 1 : 0;                            // REFLECT or SYMMETRIC
//     Nd4jLong startR = mode == 1 ? inDimSize-2 : inDimSize-1;        // REFLECT or SYMMETRIC

//     Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);

//     NDArray outSubArr0 = output(outIdx[0], true);

//     PRAGMA_OMP_PARALLEL_FOR
//     for(Nd4jLong j = 0; j < numOfSubArrs; ++j) {

//         NDArray outSubArr1   = outSubArr0(j, dimsToExclude);
//         NDArray inSubArr     = input(j, dimsToExclude);
//         NDArray outSubArrMid = outSubArr1(outIdx[1]);

//         outSubArrMid.assign(inSubArr);      // assign middle

//         if(mode == 0)  { // CONSTANT
//             if(numLeft != 0) {
//                 NDArray temp = outSubArr1(outIdx[2]);
//                 temp.assign(padValue);                        // assign left
//             }
//             if(numRight != 0) {
//                 NDArray temp = outSubArr1(outIdx[3]);
//                 temp.assign(padValue);                        // assign right
//             }
//         }
//         else {                                                              // REFLECT or SYMMETRIC

//             for(Nd4jLong k = numLeft-1, e = startL; k >= 0; --k, ++e)     // fill left side
//                 outSubArr1.t<T>(k) = inSubArr.t<T>(e);

//             for(Nd4jLong k = numLeft + inDimSize, e = startR; k < outDimSize; ++k, --e)     // fill right side
//                 outSubArr1.t<T>(k) = inSubArr.t<T>(e);
//         }
//     }

//     // ***** fill rest of outer sub-arrays ***** //
//     std::vector<Nd4jLong> outIdxInner(2, 0);
//     std::vector<Nd4jLong> outIdxOuter(2, 0);

//     for(int i = rankBorder - 1; i >= 0; --i) {

//         dimsToExclude.pop_back();

//         outIdxInner.push_back(0), outIdxInner.push_back(0);
//         outIdxOuter.push_back(0), outIdxOuter.push_back(0);

//         Nd4jLong numLeft  = paddings.e<Nd4jLong>(i, 0);
//         Nd4jLong numRight = paddings.e<Nd4jLong>(i, 1);

//         if(numLeft == 0 && numRight == 0)
//             continue;

//         Nd4jLong inDimSize  = input.sizeAt(i);
//         Nd4jLong outDimSize = output.sizeAt(i);

//         if(mode == 0) {
//             outIdxOuter[0] = 0;                   outIdxOuter[1] = numLeft;
//             outIdxInner[0] = numLeft + inDimSize; outIdxInner[1] = outDimSize;
//         }

//         startL = mode == 1 ? numLeft + 1 : numLeft;                            // REFLECT or SYMMETRIC
//         startR = mode == 1 ? numLeft + inDimSize - 2 : numLeft + inDimSize-1;      // REFLECT or SYMMETRIC

//         numOfSubArrs = ShapeUtils::getNumOfSubArrs(output.getShapeInfo(), dimsToExclude);

//         PRAGMA_OMP_PARALLEL_FOR_ARGS(firstprivate(outIdxOuter, outIdxInner))
//         for(Nd4jLong j = 0; j < numOfSubArrs; ++j) {

//             NDArray outSubArr = output(j, dimsToExclude);

//             if(mode == 0)  { // CONSTANT

//                 if(numLeft != 0) {
//                     NDArray tempO = outSubArr(outIdxOuter);
//                     tempO.assign(padValue);                              // assign left
//                 }

//                 if(numRight != 0) {
//                     NDArray tempI = outSubArr(outIdxInner);
//                     tempI.assign(padValue);                              // assign right
//                 }
//             }
//             else {                                                              // REFLECT or SYMMETRIC

//                 for(Nd4jLong k = numLeft-1, e = startL; k >= 0; --k, ++e) {    // fill left side
//                     outIdxOuter[0] = k;
//                     outIdxOuter[1] = k+1;
//                     outIdxInner[0] = e;
//                     outIdxInner[1] = e+1;
//                     NDArray outSubArrInner = outSubArr(outIdxInner);
//                     NDArray outSubArrOuter = outSubArr(outIdxOuter);
//                     outSubArrOuter.assign(outSubArrInner);
//                 }

//                 for(Nd4jLong k = numLeft + inDimSize, e = startR; k < outDimSize; ++k, --e) {    // fill right side
//                     outIdxOuter[0] = k;
//                     outIdxOuter[1] = k+1;
//                     outIdxInner[0] = e;
//                     outIdxInner[1] = e+1;
//                     NDArray outSubArrInner = outSubArr(outIdxInner);
//                     NDArray outSubArrOuter = outSubArr(outIdxOuter);
//                     outSubArrOuter.assign(outSubArrInner);
//                 }
//             }
//         }
//     }
// }

void pad(nd4j::LaunchContext * context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {
    BUILD_SINGLE_SELECTOR(input.dataType(), pad_, (mode, input, paddings, output, padValue), LIBND4J_TYPES);
}

////////////////////////////////////////////////////////////////////////
/*// initial values of inIdx, outIdx, dim must be equal to zero
template<typename T>
static void recursiveLoopForPad_(const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue ) {

    int leftOffset;
    // dimensions are array of input dimensions, it is sorted in increasing order
    // every time at the beginning we erase first element from it (not good idea to use vector for this purpose, but luckily it is small enough)
    // then we use this array for tads building, every time while recursion the number of built tads becomes bigger
    dimensions.erase(dimensions.begin());
    // build tad basing on output array, also create auxiliary arrays pointing on required output array ranges
    shape::TAD tadOut(output.getShapeInfo(), dimensions.data(), dimensions.size());
    tadOut.createTadOnlyShapeInfo();
    tadOut.createOffsets();
    auto subArrOut = NDArray(output.getBuffer(), tadOut.tadOnlyShapeInfo, output.getContext());
    auto subArr = NDArray(output.getBuffer(), tadOut.tadOnlyShapeInfo, output.getContext());
    // build tad basing on input array, also create auxiliary array pointing on required input array range
    shape::TAD tadIn(input.getShapeInfo(), dimensions.data(), dimensions.size());
    tadIn.createTadOnlyShapeInfo();
    tadIn.createOffsets();
    auto subArrIn = NDArray(input.getBuffer(), tadIn.tadOnlyShapeInfo, output.getContext());
    // these indices take into account recursion and always point to actual tads numbers
    if (input.rankOf() > 1 && output.rankOf() > 1) {// only for non-vector cases
        outIdx = outIdx * output.sizeAt(dim + 1);
        inIdx = inIdx * input.sizeAt(dim + 1);
    }
    // current input tad number, we add to it unity in a loop
    int k = -1;
    // loop through current dimension
    for(int i = 0; i < output.sizeAt(dim); ++i) {
        // corresponds to outer range (relevant indices are absent in input)
        leftOffset = paddings.e<int>(dim, 0);
        if(i < leftOffset || i >= (input.sizeAt(dim) + leftOffset))
            continue;

        // increase input tads number
        ++k;
        // recursion condition allows for the fact that tad can't reduce to scalar
        if(dim < input.rankOf() - 2)
            recursiveLoopForPad(mode, input, paddings, output, dimensions, dim + 1, inIdx + k, outIdx + i, padValue);
        else if (paddings.sizeAt(0) > dim + 1){
            leftOffset = paddings.e<int>(dim + 1, 0);
            // shift buffers pointers to actual element position
            if (output.rankOf() > 1) {
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + i]);
                subArrIn.setBuffer(reinterpret_cast<T*>(input.getBuffer()) + tadIn.tadOffsets[inIdx + i - paddings.e<int>(dim, 0)]);
            }
            else {
                subArrOut.p(i, subArrIn.e<T>(i - leftOffset));
            }
            // most inner loop, corresponds to last dim = rank-1
            switch (mode) {
                case 0:             // CONSTANT mode
                    for(int j = 0; j < subArrOut.lengthOf(); ++j)
                            if(j < leftOffset || j >= (subArrIn.lengthOf() + leftOffset) )                  // firstly fill with zeros outer ranges
                                subArrOut.p(j, (T)0.f);
                            else
                                subArrOut.p(j, subArrIn.e<T>(j - leftOffset));   // fill middle with elements of input array
                    break;

                case 1:             // REFLECT mode
                    for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side
                        subArrOut.p(leftOffset - j, subArrIn.e<T>(j));
                    for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
                        subArrOut.p(leftOffset + j, subArrIn.e<T>(j));
                    for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
                        subArrOut.p(j, subArrIn.e<T>(subArrOut.lengthOf() - j - 1));
                    break;

                case 2:             // SYMMETRIC mode
                    for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side
                        subArrOut.p(leftOffset - j, subArrIn.e<T>(j-1));
                    for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
                        subArrOut.p(leftOffset + j, subArrIn.e<T>(j));
                    for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
                        subArrOut.p(j, subArrIn.e<T>(subArrOut.lengthOf() - j));
                    break;
            }
        }
        else {

             if (mode == 0 && input.rankOf() < 2)
                 subArrOut.p(i, subArrIn.e<T>(i - leftOffset));   // fill middle with elements of input array
        }
    }
    // populate sub-array formed previously
    leftOffset = paddings.e<int>(dim,0);
    switch (mode) {
        case 0:         // CONSTANT mode
            for(int j = 1;  j <= leftOffset; ++j) {
                // fill left side with padValue
                if (output.rankOf() > 1) {
                    subArrOut.setBuffer(
                            reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset - j]);
                    subArrOut.assign(padValue);
                }
                else {
                    subArrOut.p(j - 1, padValue);
                }
            }
//            output.printIndexedBuffer("Output at");
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill left side with zeros
                if (output.rankOf() > 1) {
                    subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + j]);
                    subArrOut.assign(padValue);
                }
                else {
                    subArrOut.p(j, padValue);
                }
            }
            break;

        case 1:         // REFLECT mode
            for(int j = 1;  j <= leftOffset; ++j) {                                                     // fill left side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset + j]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset - j]);
                subArrOut.assign(&subArr);
            }
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill right side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + output.sizeAt(dim) + leftOffset - 1 - j]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + j]);
                subArrOut.assign(&subArr);
            }
            break;

        case 2:         // SYMMETRIC mode
            for(int j = 1;  j <= leftOffset; ++j) {                                                     // fill left side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset + j - 1]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + leftOffset - j]);
                subArrOut.assign(&subArr);
            }
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill right side
                subArr.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + output.sizeAt(dim) + leftOffset - j]);
                subArrOut.setBuffer(reinterpret_cast<T*>(output.getBuffer()) + tadOut.tadOffsets[outIdx + j]);
                subArrOut.assign(&subArr);
            }
            break;
    }
}
 */
/*
    void recursiveLoopForPad(const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue ) {
        BUILD_SINGLE_SELECTOR(input.dataType(), recursiveLoopForPad_, (mode, input, paddings, output, dimensions, dim, inIdx, outIdx, padValue), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void recursiveLoopForPad_, (const int mode, NDArray& input, const NDArray& paddings, NDArray& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, NDArray& padValue), LIBND4J_TYPES);

*/

////////////////////////////////////////////////////////////////////////
void invertPermutation(nd4j::LaunchContext * context, const NDArray& input, NDArray& output) {

    std::set<int> uniqueElems;
    const int length = input.lengthOf();

    for(int i = 0; i < length; ++i) {

        int elem = input.e<int>(i);

        if(!uniqueElems.insert(elem).second)        // this operation forbids us to use #pragma omp
            throw std::runtime_error("helpers::invertPermutation function: input array contains duplicates !");

        if(elem < 0 || elem > length - 1)
            throw  std::runtime_error("helpers::invertPermutation function: element of input array is out of range (0, length-1) !");

        output.p<int>(elem, i);
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void gatherND_(NDArray& input, NDArray& indices, NDArray& output) {

    const X* x = reinterpret_cast<X*>(input.getBuffer());
    const Y* y = reinterpret_cast<Y*>(indices.getBuffer());
          X* z = reinterpret_cast<X*>(output.getBuffer());

    const int xRank    = input.rankOf();
    const int yRank    = indices.rankOf();
    const int zRank    = output.rankOf();
    const int maxRank  = nd4j::math::nd4j_max<int>(yRank, nd4j::math::nd4j_max<int>(xRank, zRank));

    const Nd4jLong zLen = output.lengthOf();

    const int yLastDim = indices.sizeAt(-1);

    auto func = PRAGMA_THREADS_FOR {
        Nd4jLong coords[MAX_RANK * 3];
        for (auto i = start; i < stop; i += increment) {
            Nd4jLong *zCoordStart, *xCoordStart;

            if (yLastDim == xRank) {
                zCoordStart = coords;
                xCoordStart = coords;
            } else if (zRank >= xRank) {
                zCoordStart = coords;
                xCoordStart = coords + zRank - xRank;
            } else {
                zCoordStart = coords + xRank - zRank;
                xCoordStart = coords;
            }

            shape::index2coords(i, output.getShapeInfo(), zCoordStart);

            const auto zOffset = shape::getOffset(output.getShapeInfo(), zCoordStart);

            // last y coordinate
            uint coordToRestore;
            if (yLastDim != xRank)
                coordToRestore = static_cast<uint>(zCoordStart[yRank - 1]);

            zCoordStart[yRank - 1] = 0;
            const auto yOffset = shape::getOffset(indices.getShapeInfo(), zCoordStart);

            //restore z coordinate
            if (yLastDim != xRank)
                zCoordStart[yRank - 1] = coordToRestore;

            // construct coordinates for x
            for (uint j = 0; j < yLastDim; ++j)
                xCoordStart[j] = y[yOffset + j * indices.stridesOf()[yRank - 1]];   // last stride

            const auto xOffset = shape::getOffset(input.getShapeInfo(), xCoordStart);

            z[zOffset] = x[xOffset];
        }
    };

    samediff::Threads::parallel_tad(func, 0, zLen);
}

////////////////////////////////////////////////////////////////////////
void gatherND(nd4j::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output) {
    BUILD_DOUBLE_SELECTOR(input.dataType(), indices.dataType(), gatherND_, (input, indices, output), LIBND4J_TYPES, INDEXING_TYPES);
}


////////////////////////////////////////////////////////////////////////
template<typename T>
static void gather_(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    int axis = intArgs.size() > 0 ? intArgs[0] : 0;
    const int inputRank = input->rankOf();
    if(axis < 0)
        axis += inputRank;

    const int numOfIntArgs = intArgs.size();

    if (indices != nullptr) {

        for(int i = 0; i < indices->lengthOf(); ++i)
            if(indices->e<Nd4jLong>(i) >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: indices array contains wrong elements, each element must be smaller than corresponding dimension of input array !");

        // first case: indices consist of only one scalar
        if(indices->isScalar()) {
            if(input->rankOf() <= 1){
                //For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead, we want to get a scalar
				auto idx = indices->e<Nd4jLong>(0);
				auto scalarNDArray = input->e(idx);
                output->assign(scalarNDArray);
            } else {
                auto dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
                auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);

                auto tadArr = NDArray(reinterpret_cast<void *>(reinterpret_cast<T*>(input->getBuffer()) + tadPack.primaryOffsets()[indices->e<Nd4jLong>(0)]), tadPack.primaryShapeInfo(), output->getContext());
                output->assign(&tadArr);
			}
        }
        else if (input->rankOf() == 1 && indices->isVector()) {
            // special case
            auto func = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment)
                    output->p(e, input->e<T>(indices->e<Nd4jLong>(e)));
            };

            samediff::Threads::parallel_for(func, 0, indices->lengthOf());
        }
        else {

            std::vector<int> dimsOut(indices->rankOf());
            std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... indices->rankOf()-1
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->getShapeInfo(), dimsOut);

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i += increment) {
                    NDArray subArrOut = (*output)(i, dimsOut);
                    NDArray subArrIn = (*input)(indices->e<Nd4jLong>(i), {axis});
                    subArrOut.assign(subArrIn);
                }
            };

            samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        }
    }
    else {

        for(int i = 1; i < numOfIntArgs; ++i)
            if(intArgs[i] >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: some of input indexes is larger than corresponding shape of input array !");

        // we only allow scalar/vector case here
        if (numOfIntArgs == 2) { // scalar case
            output->assign((*input)(intArgs[1], {axis}));
        }
        else { // vector case
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->getShapeInfo(), {axis});

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i += increment) {
                    NDArray subArrOut = (*output)(i, {axis});
                    NDArray subArrIn = (*input)(intArgs[i + 1], {axis});
                    subArrOut.assign(subArrIn);
                }
            };

            samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        }
    }
}

    void gather(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
        BUILD_SINGLE_SELECTOR(input->dataType(), gather_, (input, indices, output, intArgs), LIBND4J_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
void eye(nd4j::LaunchContext * context, NDArray& output) {

    const int rank = output.rankOf();
    auto arrs = output.allTensorsAlongDimension({rank-2, rank-1});

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i += increment)
            arrs.at(i)->setIdentity();
    };

    samediff::Threads::parallel_tad(func, 0, arrs.size());
}

//////////////////////////////////////////////////////////////////////////
void scatterUpdate(nd4j::LaunchContext * context, NDArray& input, NDArray& updates, const std::vector<int>* intArgs) {

    int opCode = (*intArgs)[0];
    int dimSize = (*intArgs)[1];
    Nd4jLong e;
    Nd4jLong limg = 2 + dimSize;
    std::vector<int> tadDimensions(dimSize);
    for (e = 2; e < limg; e++)
        tadDimensions[e-2] = (*intArgs)[e];

    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input.rankOf(), tadDimensions);

    // increasing counter to skip numIndices
    e++;
    std::vector<int> indices;
    for (; e < intArgs->size(); e++)
        indices.push_back((*intArgs)[e]);

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i += increment) {
            auto inSubArr = input(indices[i], dimsToExclude, true);
            auto updSubArr = updates(i, dimsToExclude, true);

            if (inSubArr.lengthOf() != updSubArr.lengthOf())
                continue;

            switch (opCode) {
                case 0:
                    inSubArr.applyPairwiseTransform(pairwise::Add, updSubArr, inSubArr);
                    break;
                case 1:
                    inSubArr.applyPairwiseTransform(pairwise::Subtract, updSubArr, inSubArr);
                    break;
                case 2:
                    inSubArr.applyPairwiseTransform(pairwise::Multiply, updSubArr, inSubArr);
                    break;
                case 3:
                    inSubArr.applyPairwiseTransform(pairwise::Divide, updSubArr, inSubArr);
                    break;
                case 4:
                    inSubArr.applyPairwiseTransform(pairwise::ReverseSubtract, updSubArr, inSubArr);
                    break;
                case 5:
                    inSubArr.applyPairwiseTransform(pairwise::ReverseDivide, updSubArr, inSubArr);
                    break;
                case 6:
                    inSubArr.applyPairwiseTransform(pairwise::CopyPws, updSubArr, inSubArr);
                    break;
                default:
                    continue;
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, indices.size());
}


//////////////////////////////////////////////////////////////////////////
void scatterSimple(nd4j::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {

    // updates and indices have same length
    const Nd4jLong len = indices.lengthOf();

    switch (opId) {

        case 6: {   // copy
            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i += increment) {
                    auto inSubArr = input(i, dimensions);
                    inSubArr.p(indices.t<Nd4jLong>(i), updates.e(i));
                }
            };

            samediff::Threads::parallel_for(func, 0, len);
        }
            break;

        default:
            throw std::invalid_argument("helpers::scatterSimple: operation is not implemented for given id !");
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeMaxIndex_(const std::vector<NDArray*>& inArrs, NDArray& output) {

    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e += increment) {
            T max = -DataTypeUtils::max<T>();
            Nd4jLong idx = 0;

            for (int i = 0; i < numArgs; i++) {
                T v = inArrs[i]->e<T>(e);
                if (v > max) {
                    max = v;
                    idx = i;
                }
            }
            output.p(e, idx);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeMaxIndex(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
    BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), mergeMaxIndex_, (inArrs, output), LIBND4J_TYPES);
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeMax_(const std::vector<NDArray*>& inArrs, NDArray& output) {
    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e += increment) {
            T max = -DataTypeUtils::max<T>();
            for (int i = 0; i < numArgs; i++) {
                T v = inArrs[i]->e<T>(e);
                if (v > max)
                    max = v;
            }
            output.p(e, max);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeMax(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
    BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (inArrs, output), LIBND4J_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAvg_(const std::vector<NDArray*>& inArrs, NDArray& output) {
    const Nd4jLong numArgs = inArrs.size();
    const T factor = 1.f / numArgs;
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e += increment) {
            T sum = 0.;
            for (int i = 0; i < numArgs; i++) {
                T v = inArrs[i]->e<T>(e);
                sum += v;
            }
            output.p<T>(e, sum * factor);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeAvg(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
    BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (inArrs, output), LIBND4J_TYPES);
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAdd_(const std::vector<NDArray*>& inArrs, NDArray& output) {

    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e += increment) {
            T sum = (T) 0.f;
            for (int i = 0; i < numArgs; i++)
                sum += inArrs[i]->e<T>(e);

            output.p(e, sum);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}
    void mergeAdd(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (inArrs, output), LIBND4J_TYPES);
    }

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
                for (auto i = start; i < stop; i += increment) {
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
                for (auto i = start; i < stop; i += increment) {
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
void clipByNorm(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
    BUILD_SINGLE_SELECTOR(output.dataType(), clipByNorm_, (input, output, dimensions, clipNorm, isInplace), FLOAT_TYPES);
}










    template <typename T>
    static void clipByGlobalNorm_(std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        T globalNorm = 0; //NDArrayFactory::create<T>(0, inputs[0]->getContext()); //sqrt(sum([l2norm(t)**2 for t in t_list]))
//        PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(sumT : globalNorm)
        for (size_t i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto l2norm = input->reduceNumber(reduce::Norm2);
            globalNorm += l2norm.t<T>(0) * l2norm.t<T>(0);
        }

        //globalNorm.applyTransform(transform::Sqrt, nullptr, nullptr);// = nd4j::math::nd4j_sqrt(globalNorm);
        auto normS = nd4j::math::nd4j_sqrt<T,T>(globalNorm);
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
    void clipByGlobalNorm(nd4j::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (std::vector<NDArray*> const& inputs, double clipNorm, nd4j::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);

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
            for (auto i = start; i < stop; i += increment) {
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

    void clipByNormBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI /*output*/, const std::vector<int>& dimensions, const NDArray& clipNorm) {
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

    void clipByAveraged(nd4j::LaunchContext * context, NDArray& input, NDArray& output, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool isInplace) {
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

    void clipByValue(nd4j::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mirrorPad_(const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {

    // mode:  0 - REFLECT, else - SYMMETRIC
    const int reflBorder = (bool)mode ? 1 : 0;
    const int rank        = input.rankOf();
    const Nd4jLong outLen = output.lengthOf();

    if(rank <= 1) {

        const Nd4jLong inLen         = input.lengthOf();
        const auto leftSide          = paddings.e<Nd4jLong>(0);
        const auto leftSideCorrected = leftSide - reflBorder;
        const Nd4jLong len           = 2*(inLen-1) + leftSide + reflBorder;

        for(int i = 0; i < outLen; ++i) {

            if (i < leftSide)                                   // left side
                output.p(i, input.e<T>(leftSideCorrected - i));

            else if(i >= leftSide && i < leftSide + inLen)      // middle
                output.p(i, input.e<T>(i - leftSide));

            else                                                // right side
                output.p(i, input.e<T>(len - i));
        }
    }
    else {

        auto func = PRAGMA_THREADS_FOR {
            Nd4jLong inIdx[MAX_RANK];
            Nd4jLong outIdx[MAX_RANK];
            for (auto i = start; i < stop; i += increment) {
                shape::index2coords(i, output.getShapeInfo(), outIdx);

                for (int j = 0; j < rank; ++j) {
                    const Nd4jLong inLen = input.sizeAt(j);
                    const auto leftSide = paddings.e<T>(j, 0);
                    const auto leftSideCorrected = leftSide - reflBorder;
                    const Nd4jLong len = 2 * (inLen - 1) + leftSide + reflBorder;

                    if (outIdx[j] < leftSide)                                        // left side
                        inIdx[j] = leftSideCorrected - outIdx[j];

                    else if (outIdx[j] >= leftSide && outIdx[j] < leftSide + inLen)  // middle
                        inIdx[j] = outIdx[j] - leftSide;

                    else                                                            // right side
                        inIdx[j] = len - outIdx[j];
                }

                auto outOffset = shape::getOffset(output.getShapeInfo(), outIdx);
                auto inOffset = shape::getOffset(input.getShapeInfo(), inIdx);
                reinterpret_cast<T *>(output.buffer())[outOffset] = reinterpret_cast<T *>(input.getBuffer())[inOffset];
            }
        };

        samediff::Threads::parallel_for(func, 0, outLen);
    }
}

    void mirrorPad(nd4j::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_SINGLE_SELECTOR(input.dataType(), mirrorPad_, (input, paddings, output, mode), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mirrorPad_, (const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void concat_(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
    nd4j::SpecialMethods<T>::concatCpuGeneric(inArrs, output, axis);
}

    void concat(nd4j::LaunchContext * context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
        BUILD_SINGLE_SELECTOR(output.dataType(), concat_,(inArrs, output, axis), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void concat_, (const std::vector<NDArray*>& inArrs, NDArray& output, const int axis), LIBND4J_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void tileBP_(const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    T* gradIBuff      = reinterpret_cast<T*>(gradI.getBuffer());
    const T* gradOBuff      = reinterpret_cast<T*>(gradO.getBuffer());
    const Nd4jLong gradILen = gradI.lengthOf();
    const Nd4jLong gradOLen = gradO.lengthOf();  // gradOLen >= gradILen
    const Nd4jLong gradIEWS = nd4j::math::nd4j_abs<Nd4jLong>(gradI.ews());
    const Nd4jLong gradOEWS = gradO.ews();

    // initial zeroing of gradI content
    if(gradIEWS == 1)
        memset(gradIBuff, 0, gradILen * sizeof(T));
    else {
        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i = 0; i < gradILen * gradIEWS; i += gradIEWS)
            gradIBuff[i] = static_cast<T>(0.f);
    }


    if(gradO.ordering() == 'c' && gradOEWS == 1) {

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            auto idx = shape::subArrayIndex(i, gradO.getShapeInfo(), gradI.getShapeInfo());
            gradI.p(idx, gradI.e<T>(idx) + gradOBuff[i]);
        }
    }
    else if(gradO.ordering() == 'c' && gradOEWS > 1) {

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            auto idx = shape::subArrayIndex(i, gradO.getShapeInfo(), gradI.getShapeInfo());
            gradI.p(idx, gradI.e<T>(idx) + gradOBuff[i * gradOEWS]);
        }
    }
    else {

        //PRAGMA_OMP_PARALLEL_FOR_SIMD
        for(Nd4jLong i=0;  i<gradOLen; ++i) {

            auto fidx = shape::subArrayIndex(i, gradO.getShapeInfo(), gradI.getShapeInfo());
            gradI.p(fidx, gradI.e<T>(fidx) + gradOBuff[shape::getIndexOffset(i, gradO.getShapeInfo())]);
        }
    }
}

void tileBP(nd4j::LaunchContext * context, const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps) {
    BUILD_SINGLE_SELECTOR(gradI.dataType(), tileBP_, (gradO, gradI, reps), FLOAT_TYPES);
}


BUILD_SINGLE_TEMPLATE(template void tileBP_, (const NDArray& gradO /*input*/, NDArray& gradI /*output*/, const std::vector<Nd4jLong> reps), FLOAT_TYPES);





}
}
}
