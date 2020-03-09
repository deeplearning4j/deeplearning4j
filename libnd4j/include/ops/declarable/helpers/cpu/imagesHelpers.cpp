/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
// @author AbdelRauf    (rauf@konduit.ai)
//

#include <ops/declarable/helpers/adjust_hue.h>
#include <ops/declarable/helpers/imagesHelpers.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void rgbToGrs_(const NDArray& input, NDArray& output, const int dimC) {

    const T* x = input.bufferAsT<T>();
    T* z = output.bufferAsT<T>();
    const int rank = input.rankOf();

    if(dimC == rank - 1 && 'c' == input.ordering() && 1 == input.ews() &&
        'c' == output.ordering() && 1 == output.ews()){

        auto func = PRAGMA_THREADS_FOR{
             for (auto i = start; i < stop; i++) {
                 const auto xStep = i*3;
                 z[i] = 0.2989f*x[xStep] + 0.5870f*x[xStep + 1] + 0.1140f*x[xStep + 2];
             }
        };

        sd::Threads::parallel_for(func, 0, output.lengthOf(), 1);
        return;
    }

    auto func = PRAGMA_THREADS_FOR{

         Nd4jLong coords[MAX_RANK];
         for (auto i = start; i < stop; i++) {
             shape::index2coords(i, output.getShapeInfo(), coords);
             const auto zOffset = shape::getOffset(output.getShapeInfo(), coords);
             const auto xOffset0 =  shape::getOffset(input.getShapeInfo(), coords);
             const auto xOffset1 = xOffset0 + input.strideAt(dimC);
             const auto xOffset2 = xOffset1 + input.strideAt(dimC);
             z[zOffset] = 0.2989f*x[xOffset0] + 0.5870f*x[xOffset1] + 0.1140f*x[xOffset2];
         }
     };

     sd::Threads::parallel_for(func, 0, output.lengthOf(), 1);
     return;
}

void transformRgbGrs(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input.dataType(), rgbToGrs_, (input, output, dimC), NUMERIC_TYPES);
}

template <typename T, typename Op>
FORCEINLINE static void rgbToFromYuv_(const NDArray& input, NDArray& output, const int dimC, Op op) {

    const T* x = input.bufferAsT<T>();
    T* z = output.bufferAsT<T>();
    const int rank = input.rankOf();
    bool bSimple = (dimC == rank - 1 && 'c' == input.ordering() && 1 == input.ews() &&
                     'c' == output.ordering() && 1 == output.ews());
    
    if (bSimple) {
        
        auto func = PRAGMA_THREADS_FOR{
            for (auto i = start; i < stop; i += increment) {
                op(x[i], x[i + 1], x[i + 2], z[i], z[i + 1], z[i + 2]);
            }
        };

        sd::Threads::parallel_for(func, 0, input.lengthOf(), 3);
        return;
    }

    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), dimC);
    auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), dimC);

    const Nd4jLong numOfTads = packX.numberOfTads();
    const Nd4jLong xDimCstride = input.stridesOf()[dimC];
    const Nd4jLong zDimCstride = output.stridesOf()[dimC];

    auto func = PRAGMA_THREADS_FOR{
        for (auto i = start; i < stop; i++) {
            const T* xTad = x + packX.platformOffsets()[i];
            T* zTad = z + packZ.platformOffsets()[i];
            op(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
        }
    };

    sd::Threads::parallel_tad(func, 0, numOfTads);
    return;
}

template <typename T>
FORCEINLINE static void rgbYuv_(const NDArray& input, NDArray& output, const int dimC) {
    auto op = sd::ops::helpers::rgbYuv<T>;
    return rgbToFromYuv_<T>(input, output, dimC, op);
}

void transformRgbYuv(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input.dataType(), rgbYuv_, (input, output, dimC), FLOAT_TYPES);
}

template <typename T>
FORCEINLINE static void yuvRgb_(const NDArray& input, NDArray& output, const int dimC) {
    auto op = sd::ops::helpers::yuvRgb<T>;
    return rgbToFromYuv_<T>(input, output, dimC, op);
}

void transformYuvRgb(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input.dataType(), yuvRgb_, (input, output, dimC), FLOAT_TYPES);
}

template <typename T, typename Op>
FORCEINLINE static void tripleTransformer(const NDArray* input, NDArray* output, const int dimC, Op op) {

    const int rank = input->rankOf();

    const T* x = input->bufferAsT<T>();
    T* z = output->bufferAsT<T>();

    if (dimC == rank - 1 && input->ews() == 1 && output->ews() == 1 && input->ordering() == 'c' && output->ordering() == 'c') {

        auto func = PRAGMA_THREADS_FOR{
            for (auto i = start; i < stop; i += increment) {
                op(x[i], x[i + 1], x[i + 2], z[i], z[i + 1], z[i + 2]);
            }
        };

        sd::Threads::parallel_for(func, 0, input->lengthOf(), 3);
    }
    else {
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimC);
        auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimC);

        const Nd4jLong numOfTads = packX.numberOfTads();
        const Nd4jLong xDimCstride = input->stridesOf()[dimC];
        const Nd4jLong zDimCstride = output->stridesOf()[dimC];

        auto func = PRAGMA_THREADS_FOR{
            for (auto i = start; i < stop; i++) {
                const T* xTad = x + packX.platformOffsets()[i];
                T* zTad = z + packZ.platformOffsets()[i];
                op(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);

            }
        };

        sd::Threads::parallel_tad(func, 0, numOfTads);
    }
}


template <typename T>
FORCEINLINE static void tripleTransformer(const NDArray* input, NDArray* output, const int dimC ,  T (&tr)[3][3] ) {

    const int rank = input->rankOf();

    const T* x = input->bufferAsT<T>();
    T* z = output->bufferAsT<T>();
    // TODO: Use tensordot or other optimizied helpers to see if we can get better performance. 

    if (dimC == rank - 1 && input->ews() == 1 && output->ews() == 1 && input->ordering() == 'c' && output->ordering() == 'c') {

        auto func = PRAGMA_THREADS_FOR{
            for (auto i = start; i < stop; i += increment) { 
                //simple M*v //tr.T*v.T // v * tr  //rule: (AB)' =B'A'
                // v.shape (1,3) row vector
                T x0, x1, x2;
                x0 = x[i]; //just additional hint
                x1 = x[i + 1];
                x2 = x[i + 2];
                z[i]   = x0 * tr[0][0] + x1 * tr[1][0] + x2 * tr[2][0];
                z[i+1] = x0 * tr[0][1] + x1 * tr[1][1] + x2 * tr[2][1];
                z[i+2] = x0 * tr[0][2] + x1 * tr[1][2] + x2 * tr[2][2];
                
            }
        };

        sd::Threads::parallel_for(func, 0, input->lengthOf(), 3);
    }
    else {
        auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimC);
        auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimC);

        const Nd4jLong numOfTads = packX.numberOfTads();
        const Nd4jLong xDimCstride = input->stridesOf()[dimC];
        const Nd4jLong zDimCstride = output->stridesOf()[dimC];

        auto func = PRAGMA_THREADS_FOR{
            for (auto i = start; i < stop; i++) {
                const T* xTad = x + packX.platformOffsets()[i];
                T* zTad = z + packZ.platformOffsets()[i];
                //simple M*v //tr.T*v
                T x0, x1, x2;
                x0 = xTad[0];
                x1 = xTad[xDimCstride];
                x2 = xTad[2 * xDimCstride];
                zTad[0]               = x0 * tr[0][0] + x1 * tr[1][0] + x2 * tr[2][0];
                zTad[zDimCstride]     = x0 * tr[0][1] + x1 * tr[1][1] + x2 * tr[2][1];
                zTad[2 * zDimCstride] = x0 * tr[0][2] + x1 * tr[1][2] + x2 * tr[2][2];

            }
        };

        sd::Threads::parallel_tad(func, 0, numOfTads);
    }
}




template <typename T>
FORCEINLINE static void hsvRgb(const NDArray* input, NDArray* output, const int dimC) {
    auto op = sd::ops::helpers::hsvToRgb<T>;
    return tripleTransformer<T>(input, output, dimC, op);
}

template <typename T>
FORCEINLINE static void rgbHsv(const NDArray* input, NDArray* output, const int dimC) {
    auto op = sd::ops::helpers::rgbToHsv<T>;
    return tripleTransformer<T>(input, output, dimC, op);
}


template <typename T>
FORCEINLINE static void rgbYiq(const NDArray* input, NDArray* output, const int dimC) {
    T arr[3][3] = {
         { (T)0.299,  (T)0.59590059,  (T)0.2115 },
         { (T)0.587, (T)-0.27455667,  (T)-0.52273617 },
         { (T)0.114, (T)-0.32134392,  (T)0.31119955 }
        };
    return tripleTransformer<T>(input, output, dimC, arr);
}

template <typename T>
FORCEINLINE static void yiqRgb(const NDArray* input, NDArray* output, const int dimC) {
    //TODO: this operation does not use the clamp operation, so there is a possibility being out of range.
    //Justify that it  will not be out of range for images data
    T arr[3][3] = {
        { (T)1,  (T)1,  (T)1 },
        { (T)0.95598634, (T)-0.27201283, (T)-1.10674021 },
        { (T)0.6208248, (T)-0.64720424, (T)1.70423049 }
    };
    return tripleTransformer<T>(input, output, dimC, arr);
}



void transformHsvRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input->dataType(), hsvRgb, (input, output, dimC), FLOAT_TYPES);
}

void transformRgbHsv(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input->dataType(), rgbHsv, (input, output, dimC), FLOAT_TYPES);
}

void transformYiqRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input->dataType(), yiqRgb, (input, output, dimC), FLOAT_TYPES);
}

void transformRgbYiq(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input->dataType(), rgbYiq, (input, output, dimC), FLOAT_TYPES);
}


}
}
}