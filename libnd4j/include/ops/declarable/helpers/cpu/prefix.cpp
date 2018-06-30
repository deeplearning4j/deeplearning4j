//
//  @author raver119@gmail.com
//

#include <ops/ops.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <NDArrayFactory.h>
#include <ops/declarable/helpers/prefix.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T, typename OpName>
            void _prefix(T* x, Nd4jLong* xShapeInfo, T* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse) {
                auto length = shape::length(xShapeInfo);

                T prevSum = OpName::startingValue();
                T sum = prevSum;
                                
                if (reverse) {
                    if (shape::elementWiseStride(xShapeInfo) == 1 && shape::elementWiseStride(zShapeInfo) == 1 &&
                        shape::order(xShapeInfo) == 'c' && shape::order(zShapeInfo) == 'c') {

                        for (Nd4jLong e = length - 1; e >= 0; --e) {
                            sum = OpName::op(sum, x[e]);
                            if (!exclusive)
                                prevSum = sum;

                            z[e] = prevSum;

                            prevSum = sum;
                        }
                    } else {
                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong zCoord[MAX_RANK];

                        int xRank = shape::rank(xShapeInfo);
                        int zRank = shape::rank(zShapeInfo);

                        auto xShape = shape::shapeOf(xShapeInfo);
                        auto zShape = shape::shapeOf(zShapeInfo);

                        auto xStride = shape::stride(xShapeInfo);
                        auto zStride = shape::stride(zShapeInfo);

                        for (Nd4jLong e = length - 1; e >= 0; --e) {
                            shape::ind2subC(xRank, xShape, e, length, xCoord);
                            shape::ind2subC(zRank, zShape, e, length, zCoord);

                            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                            sum = OpName::op(sum, x[xOffset]);
                            if (!exclusive)
                                prevSum = sum;

                            z[zOffset] = prevSum;

                            prevSum = sum;
                        }
                    }
                } else {
                    if (shape::elementWiseStride(xShapeInfo) == 1 && shape::elementWiseStride(zShapeInfo) == 1 &&
                        shape::order(xShapeInfo) == 'c' && shape::order(zShapeInfo) == 'c') {                        

                        for (int e = 0; e < length; e++) {
                            sum = OpName::op(sum, x[e]);

                            if (!exclusive)
                                prevSum = sum;

                            z[e] = prevSum;

                            prevSum = sum;
                        }
                    } else {
                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong zCoord[MAX_RANK];

                        auto xRank = shape::rank(xShapeInfo);
                        auto zRank = shape::rank(zShapeInfo);

                        auto xShape = shape::shapeOf(xShapeInfo);
                        auto zShape = shape::shapeOf(zShapeInfo);

                        auto xStride = shape::stride(xShapeInfo);
                        auto zStride = shape::stride(zShapeInfo);

                        for (int e = 0; e < length; e++) {
                            shape::ind2subC(xRank, xShape, e, length, xCoord);
                            shape::ind2subC(zRank, zShape, e, length, zCoord);

                            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                            sum = OpName::op(sum, x[xOffset]);

                            if (!exclusive)
                                prevSum = sum;

                            z[zOffset] = prevSum;

                            prevSum = sum;
                        }
                    }
                }
            };

            template <typename T, typename OpName>
            void _prefix(NDArray<T>* x, NDArray<T>* z, std::vector<int>& dims, bool exclusive, bool reverse) {
                auto xTads = NDArrayFactory<T>::allTensorsAlongDimension(x, dims);
                auto zTads = NDArrayFactory<T>::allTensorsAlongDimension(z, dims);
                auto t = xTads->size();

// #pragma omp parallel for schedule(guided)
                for (int e = 0; e < t; e++) {
                    auto tx = xTads->at(e);
                    auto tz = zTads->at(e);

                    _prefix<T, OpName>(tx->buffer(), tx->shapeInfo(), tz->buffer(), tz->shapeInfo(), exclusive, reverse);
                }

                delete xTads;
                delete zTads;
            };

            template void _prefix<float, simdOps::Add<float>>(float* x, Nd4jLong* xShapeInfo, float* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
            template void _prefix<float16, simdOps::Add<float16>>(float16* x, Nd4jLong* xShapeInfo, float16* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
            template void _prefix<double, simdOps::Add<double>>(double* x, Nd4jLong* xShapeInfo, double* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);

            template void _prefix<float, simdOps::Multiply<float>>(float* x, Nd4jLong* xShapeInfo, float* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
            template void _prefix<float16, simdOps::Multiply<float16>>(float16* x, Nd4jLong* xShapeInfo, float16* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);
            template void _prefix<double, simdOps::Multiply<double>>(double* x, Nd4jLong* xShapeInfo, double* z, Nd4jLong* zShapeInfo, bool exclusive, bool reverse);


            template void _prefix<float, simdOps::Add<float>>(NDArray<float>* x, NDArray<float>* z, std::vector<int>& dims, bool exclusive, bool reverse);
            template void _prefix<float16, simdOps::Add<float16>>(NDArray<float16>* x, NDArray<float16>* z, std::vector<int>& dims, bool exclusive, bool reverse);
            template void _prefix<double, simdOps::Add<double>>(NDArray<double>* x, NDArray<double>* z, std::vector<int>& dims, bool exclusive, bool reverse);

            template void _prefix<float, simdOps::Multiply<float>>(NDArray<float>* x, NDArray<float>* z, std::vector<int>& dims, bool exclusive, bool reverse);
            template void _prefix<float16, simdOps::Multiply<float16>>(NDArray<float16>* x, NDArray<float16>* z, std::vector<int>& dims, bool exclusive, bool reverse);
            template void _prefix<double, simdOps::Multiply<double>>(NDArray<double>* x, NDArray<double>* z, std::vector<int>& dims, bool exclusive, bool reverse);
        }
    }
}