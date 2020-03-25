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

            int zCoords[MAX_RANK], xCoords[MAX_RANK];

            for (auto i = start; i < stop; i++) {

                shape::index2coordsCPU(start, i, output.getShapeInfo(), zCoords);
                const auto zOffset = shape::getOffset(output.getShapeInfo(), zCoords);

                memcpy(xCoords, zCoords, rank * sizeof(int));

                bool within = true;

                for (int j = rankMinusOne; j >= 0; --j) {

                    if (xShape[j] == zShape[j])
                        continue;

                    const auto left = paddings.e<Nd4jLong>(j, 0);

                    if (zCoords[j] < left || zCoords[j] >= left + xShape[j]) {
                        within = false;
                        break;
                    }
                    else
                        xCoords[j] = zCoords[j] - left;
                }

                if (within)
                    z[zOffset] = x[shape::getOffset(input.getShapeInfo(), xCoords)];
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

            int zCoords[MAX_RANK], xCoords[MAX_RANK];

            for (auto i = start; i < stop; i++) {

                shape::index2coordsCPU(start, i, output.getShapeInfo(), zCoords);
                const auto zOffset = shape::getOffset(output.getShapeInfo(), zCoords);

                memcpy(xCoords, zCoords, rank * sizeof(int));

                for (int j = rankMinusOne; j >= 0; --j) {

                    if (xShape[j] == zShape[j])
                        continue;

                    xCoords[j] = zCoords[j] - paddings.e<Nd4jLong>(j, 0);                             // are ready to fill middle (within input dimension range)

                    if (xCoords[j] < 0)
                        xCoords[j] = -xCoords[j] - shift1;                // means fill from left
                    else if (xCoords[j] >= xShape[j])
                        xCoords[j] = 2 * xShape[j] - xCoords[j] - shift2; // means fill from right
                }

                const auto xOffset = shape::getOffset(input.getShapeInfo(), xCoords);
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

void pad(sd::LaunchContext * context, const int mode, const NDArray& input, const NDArray& paddings, NDArray& output, NDArray const& padValue) {
    BUILD_SINGLE_SELECTOR(input.dataType(), pad_, (mode, input, paddings, output, padValue), LIBND4J_TYPES);
}

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

            int inIdx[MAX_RANK], outIdx[MAX_RANK];

            for (auto i = start; i < stop; i++) {

                shape::index2coordsCPU(start, i, output.getShapeInfo(), outIdx);

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

    void mirrorPad(sd::LaunchContext * context, const NDArray& input, const NDArray& paddings, NDArray& output, const int mode) {
        BUILD_SINGLE_SELECTOR(input.dataType(), mirrorPad_, (input, paddings, output, mode), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void mirrorPad_, (const NDArray& input, const NDArray& paddings, NDArray& output, const int mode), LIBND4J_TYPES);


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

}
}
}
