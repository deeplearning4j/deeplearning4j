//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/broadcasting.h>
#include <loops/legacy_ops.h>

namespace functions {
    namespace broadcast {

        template <typename T>
        void Broadcast<T>::exec(const int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *y,
                             int *yShapeInfo,
                             T *result,
                             int *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset,
                             int *tadShapeInfoZ,
                             Nd4jIndex *tadOffsetZ) {
            DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               y,
                                               yShapeInfo,
                                               result,
                                               resultShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset,
                                               tadShapeInfoZ,
                                               tadOffsetZ), BROADCAST_OPS);
        }

        template <typename T>
        template<typename OpType>
        void Broadcast<T>::exec(T *x,
                             int *xShapeInfo,
                             T *y,
                             int *yShapeInfo,
                             T *result,
                             int *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset,
                             int *tadShapeInfoZ,
                             Nd4jIndex *tadOffsetZ) {


                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                int *tadShapeShapeInfo = tadShapeInfo;
                Nd4jIndex *tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    tadShapeShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }

                //int *resultStride = shape::stride(tadShapeShapeInfo);
                int tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                int yStride = shape::elementWiseStride(yShapeInfo);
                int tads = shape::length(xShapeInfo) / tadLength;

                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeShapeInfo;
                    tadOffsetZ = tadOffsets;
                }

                int zEWS = shape::elementWiseStride(tadShapeInfoZ);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(_threads) if (_threads > 1) proc_bind(AFFINITY) default(shared)
                for (int i = 0; i < tads; i++) {
                    Nd4jIndex offset = tadOffsets[i];
                    Nd4jIndex offsetZ = tadOffsetZ[i];
//                    printf("Tad: [%i]; Offset: [%lld]; OffsetZ: [%lld];\n", i, offset, offsetZ);


                    if (tadEWS > 0 && yStride > 0 && zEWS > 0 && dimensionLength == 1) {
                        T *oRes = result + offsetZ;
                        T *oX = x + offset;

                        if (tadEWS == 1 && yStride == 1 && zEWS == 1) {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oRes[f] = OpType::op(oX[f], y[f]);
                            }
                        } else {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oRes[f * zEWS] = OpType::op(oX[f * tadEWS], y[f * yStride]);
                            }
                        }
                    }
                    else {
                        int *zShape = shape::shapeOf(tadShapeInfoZ);
                        int *zStride = shape::stride(tadShapeInfoZ);
                        int zRank = shape::rank(tadShapeInfoZ);

                        int *xShape = shape::shapeOf(tadShapeShapeInfo);
                        int *xStride = shape::stride(tadShapeShapeInfo);
                        int xRank = shape::rank(tadShapeShapeInfo);

                        int *yShape = shape::shapeOf(yShapeInfo);
                        int *yStride = shape::stride(yShapeInfo);
                        int yRank = shape::rank(yShapeInfo);

                        int xCoord[MAX_RANK];
                        int yCoord[MAX_RANK];
                        int zCoord[MAX_RANK];


                        // TODO: cover this codebranch with tests
                        // all this stuff already happens within thread
                        for (int f = 0; f < tadLength; f++) {
                            if (shape::order(tadShapeShapeInfo) == 'c') {
                                shape::ind2subC(xRank, xShape, f, xCoord);
                                shape::ind2subC(yRank, yShape, f, yCoord);
                            } else {
                                shape::ind2sub(xRank, xShape, f, xCoord);
                                shape::ind2sub(yRank, yShape, f, yCoord);
                            }

                            if (shape::order(tadShapeInfoZ) == 'c')
                                shape::ind2subC(zRank, zShape, f, zCoord);
                            else
                                shape::ind2sub(zRank, zShape, f, zCoord);

                            Nd4jIndex xOffset = shape::getOffset(offset, xShape, xStride, xCoord, xRank);
                            Nd4jIndex zOffset = shape::getOffset(offsetZ, zShape, zStride, zCoord, zRank);
                            Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                            result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                        }
                    }
                }

                if (tad != nullptr)
                    delete tad;
        }

        template class ND4J_EXPORT Broadcast<float>;
        template class ND4J_EXPORT Broadcast<float16>;
        template class ND4J_EXPORT Broadcast<double>;

        BUILD_CALL_1(template void Broadcast<float>::exec, float, (float*, int*, float*, int*, float*, int*, int*, int, int*, long long*, int*, Nd4jIndex*), BROADCAST_OPS)
        BUILD_CALL_1(template void Broadcast<float16>::exec, float16, (float16*, int*, float16*, int*, float16*, int*, int*, int, int*, long long*, int*, Nd4jIndex*), BROADCAST_OPS)
        BUILD_CALL_1(template void Broadcast<double>::exec, double, (double*, int*, double*, int*, double*, int*, int*, int, int*, long long*, int*, Nd4jIndex*), BROADCAST_OPS)
    }
}