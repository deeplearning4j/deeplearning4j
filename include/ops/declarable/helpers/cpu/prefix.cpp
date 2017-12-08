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
            void _prefix(T* x, int* xShapeInfo, T* z, int* zShapeInfo) {
                auto length = shape::length(xShapeInfo);

                if (shape::elementWiseStride(xShapeInfo) == 1 && shape::elementWiseStride(zShapeInfo) == 1 && shape::order(xShapeInfo) == 'c' && shape::order(zShapeInfo) == 'c') {
                    T sum = (T) 0;                    
#pragma omp simd                    
                    for (int e = 0; e < length; e++) {
                        sum = OpName::op(sum, x[e]);
                        z[e] = sum;
                    }
                } else {
                    int xCoord[MAX_RANK];
                    int zCoord[MAX_RANK];
                    T sum = (T) 0;  

                    int xRank = shape::rank(xShapeInfo);
                    int zRank = shape::rank(zShapeInfo);

                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *zShape = shape::shapeOf(zShapeInfo);

                    int *xStride = shape::stride(xShapeInfo);
                    int *zStride = shape::stride(zShapeInfo);

                    for (int e = 0; e < length; e++) {
                        shape::ind2subC(xRank, xShape, e, xCoord);
                        shape::ind2subC(zRank, zShape, e, zCoord);

                        Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        Nd4jIndex zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                        sum = OpName::op(sum, x[xOffset]);
                        z[zOffset] = sum;
                    }
                }
            };

            template <typename T, typename OpName>
            void _prefix(NDArray<T>* x, NDArray<T>* z, std::vector<int>& dims) {
                auto xTads = NDArrayFactory<T>::allTensorsAlongDimension(x, dims);
                auto zTads = NDArrayFactory<T>::allTensorsAlongDimension(z, dims);
                int t = xTads->size();

#pragma omp parallel for schedule(guided)
                for (int e = 0; e < t; e++) {
                    auto tx = xTads->at(e);
                    auto tz = zTads->at(e);

                    _prefix<T, OpName>(tx->buffer(), tx->shapeInfo(), tz->buffer(), tz->shapeInfo());
                }

                delete xTads;
                delete zTads;
            };

            template void _prefix<float, simdOps::Add<float>>(float* x, int* xShapeInfo, float* z, int* zShapeInfo);
            template void _prefix<float16, simdOps::Add<float16>>(float16* x, int* xShapeInfo, float16* z, int* zShapeInfo);
            template void _prefix<double, simdOps::Add<double>>(double* x, int* xShapeInfo, double* z, int* zShapeInfo);

            template void _prefix<float, simdOps::Multiply<float>>(float* x, int* xShapeInfo, float* z, int* zShapeInfo);
            template void _prefix<float16, simdOps::Multiply<float16>>(float16* x, int* xShapeInfo, float16* z, int* zShapeInfo);
            template void _prefix<double, simdOps::Multiply<double>>(double* x, int* xShapeInfo, double* z, int* zShapeInfo);


            template void _prefix<float, simdOps::Add<float>>(NDArray<float>* x, NDArray<float>* z, std::vector<int>& dims);
            template void _prefix<float16, simdOps::Add<float16>>(NDArray<float16>* x, NDArray<float16>* z, std::vector<int>& dims);
            template void _prefix<double, simdOps::Add<double>>(NDArray<double>* x, NDArray<double>* z, std::vector<int>& dims);

            template void _prefix<float, simdOps::Multiply<float>>(NDArray<float>* x, NDArray<float>* z, std::vector<int>& dims);
            template void _prefix<float16, simdOps::Multiply<float16>>(NDArray<float16>* x, NDArray<float16>* z, std::vector<int>& dims);
            template void _prefix<double, simdOps::Multiply<double>>(NDArray<double>* x, NDArray<double>* z, std::vector<int>& dims);
        }
    }
}