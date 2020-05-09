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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <ops/declarable/helpers/qr.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void matrixMinorKernel(T* outBuffer, Nd4jLong* outShape, T* inBuffer, Nd4jLong* inShape, Nd4jLong column, Nd4jLong rows, Nd4jLong columns) {
//        auto tid = threadIdx.x + blockDim.x * blockIdx.x;
//        auto step = blockDim.x * gridDim.x;
//        if (threadIdx.x == 0) {
//            for (auto i = tid; i < column; i += step) {
//                Nd4jLong diagPos[] = {i, i};
//                auto zIndex = shape::getOffset(outShape, diagPos);
//                outBuffer[zIndex] = T(1.f);
//            }
//        }
//        __syncthreads();

        for (auto i = blockIdx.x; i < rows; i += gridDim.x)
            for (auto j = threadIdx.x; j < columns; j += blockDim.x) {
                Nd4jLong pos[] = {i,j};
                auto zIndex = shape::getOffset(outShape, pos);
                auto xIndex = shape::getOffset(inShape, pos);
                if (i < column || j < column) {
                    outBuffer[zIndex] = i != j?T(0.f):T(1.f);
                }
                else
                    outBuffer[zIndex] = inBuffer[xIndex]; //m.t<T>(i,j) = in.t<T>(i,j);
            }


    }

    template <typename T>
    NDArray matrixMinor(LaunchContext* context, NDArray& in, Nd4jLong col) {
        NDArray m = in.ulike();
        m.setIdentity();
        m({col, m.rows(), col, m.columns()}).assign(in({col, m.rows(), col, m.columns()}));

//        auto stream = context->getCudaStream();
//        matrixMinorKernel<T><<<128, 128, 256, *stream>>>(m.dataBuffer()->specialAsT<T>(), m.specialShapeInfo(),
//        matrixMinorKernel<T><<<128, 128, 256, *stream>>>(m.dataBuffer()->specialAsT<T>(), m.specialShapeInfo(),
//                reinterpret_cast<T*>(in.specialBuffer()), in.specialShapeInfo(), col, in.rows(), in.columns());
//
        m.tickWriteDevice();
        return m;
    }

/* m = I - v v^T */
    template <typename T>
    static __global__ void vmulKernel(T* resBuf, const Nd4jLong* resShape, T const* vBuff, Nd4jLong const* vShape, Nd4jLong n) {
        for (auto i = blockIdx.x; i < n; i += gridDim.x)
            for (auto j = threadIdx.x; j < n; j += blockDim.x) {
                Nd4jLong posR[] = {i, j};
                auto indexR = shape::getOffset(resShape, posR);
                auto indexX = shape::getIndexOffset(i, vShape);
                auto indexY = shape::getIndexOffset(j, vShape);

                resBuf[indexR] = T(-2.f) * vBuff[indexX] * vBuff[indexY] + (i != j?T(0.f):T(1.f));
            }
    }

    template <typename T>
    NDArray vmul(LaunchContext* context, NDArray const& v, int n)
    {
        NDArray res('c', {n,n}, v.dataType(), context); // x = matrix_new(n, n);

        auto stream = context->getCudaStream();
        vmulKernel<T><<<128, 128, 128, *stream>>>(res.dataBuffer()->specialAsT<T>(), res.specialShapeInfo(),
                reinterpret_cast<T const*>(v.specialBuffer()), v.specialShapeInfo(), n);
        return res;
    }

    template <typename T>
    static bool diagonalIsPositive(NDArray* matrix, Nd4jLong k) {
        T hVal;
        Nd4jLong pos[] = {k, k};
        auto shift = shape::getOffset(matrix->shapeInfo(), pos);
        cudaMemcpy(&hVal, matrix->specialBuffer(), sizeof(T), cudaMemcpyDeviceToHost);
        return hVal > T(0.f);
    }

    template <typename T>
    void qrSingle(LaunchContext* context, NDArray* matrix, NDArray* Q, NDArray* R, bool const fullMatricies) {
        Nd4jLong M = matrix->sizeAt(0);
        Nd4jLong N = matrix->sizeAt(1);
        auto resQ = fullMatricies?Q->ulike():NDArrayFactory::create<T>(matrix->ordering(), {M,M}, Q->getContext());
        auto resR = fullMatricies?R->ulike():matrix->ulike();
        std::vector<NDArray> q(M);
        NDArray z = *matrix;
        NDArray e('c', {M}, DataTypeUtils::fromT<T>(), context); // two internal buffers and scalar for squared norm
        for (auto k = 0; k < N && k < M - 1; k++) { // loop for columns, but not further then row number
            e.nullify();
            z = matrixMinor<T>(context, z, k); // minor computing for current column with given matrix z (initally is a input matrix)

            auto currentColumn = z({0, 0, k, k + 1}); // retrieve k column from z to x buffer
            auto norm = currentColumn.reduceAlongDimension(reduce::Norm2, {0});
            if (diagonalIsPositive<T>(matrix, k)) //matrix->t<T>(k,k) > T(0.f)) // negate on positive matrix diagonal element
                norm.applyTransform(transform::Neg, norm); // *= -1.f;//-norm.t<T>(0);

            e.p(k, norm); // e - is filled by 0 vector except diagonal element (filled by 1)
            e += currentColumn; // e[i] = x[i] + a * e[i] for each i from 0 to n - 1
            auto normE = e.reduceAlongDimension(reduce::Norm2, {0});
            e /= normE;
            q[k] = vmul<T>(context, e, M);
            auto qQ = z.ulike();
            MmulHelper::matmul(&q[k], &z, &qQ, false, false);
            z = std::move(qQ);
        }
        resQ.assign(q[0]); //
//        MmulHelper::matmul(&q[0], matrix, &resR, false, false);
        for (int i = 1; i < N && i < M - 1; i++) {
            auto tempResQ = resQ;
            MmulHelper::matmul(&q[i], &resQ, &tempResQ, false, false);
            resQ = std::move(tempResQ);
        }
        MmulHelper::matmul(&resQ, matrix, &resR, false, false);
        // resR *= -1.f;
        resQ.transposei();

        if (fullMatricies) {
            Q->assign(resQ);
            R->assign(resR);
        }
        else {
            Q->assign(resQ({0, 0, 0, N}));
            R->assign(resR({0, N, 0, 0}));
        }
    }

    template <typename T>
    void qr_(LaunchContext* context, NDArray const* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
        Nd4jLong lastDim = input->rankOf() - 1;
        Nd4jLong preLastDim = input->rankOf() - 2;

        NDArray::prepareSpecialUse({outputQ, outputR}, {input});
        ResultSet listOutQ(outputQ->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        ResultSet listOutR(outputR->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        ResultSet listInput(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        auto start = 0;
        auto stop = listInput.size();
        auto increment = 1;

        for (auto batch = start; batch < stop; batch += increment) {
            //qr here
            qrSingle<T>(context, listInput.at(batch), listOutQ.at(batch), listOutR.at(batch), fullMatricies);
        }
        NDArray::registerSpecialUse({outputQ, outputR}, {input});
    }

    void qr(sd::LaunchContext* context, NDArray const* input, NDArray* outputQ, NDArray* outputR, bool const fullMatricies) {
        BUILD_SINGLE_SELECTOR(input->dataType(), qr_, (context, input, outputQ, outputR, fullMatricies), FLOAT_TYPES);
    }

}
}
}
