//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/broadcasting.h>
#include <loops/legacy_ops.h>


template <typename T, typename OpClass>
__device__ void broadcastSimpleGeneric(
		T *x,
		Nd4jLong *xShapeInfo,
		T *y,
		Nd4jLong *yShapeInfo,
		T *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {


	functions::broadcast::Broadcast<T>::template transformCuda<OpClass>(
			x,
			xShapeInfo,
			y,
			yShapeInfo,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			NULL,
			tadOnlyShapeInfo,
			tadOffsets,
			tadOnlyShapeInfoZ,
			tadOffsetsZ);
}

// broadcast kernel call
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float, INPUT(float *x, Nd4jLong *xShapeInfo, float *y, Nd4jLong *yShapeInfo, float *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, double, INPUT(double *x, Nd4jLong *xShapeInfo, double *y, Nd4jLong *yShapeInfo, double *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float16, INPUT(float16 *x, Nd4jLong *xShapeInfo, float16 *y, Nd4jLong *yShapeInfo, float16 *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))


namespace functions {
    namespace broadcast {

        template <>
        __host__ void Broadcast<float>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, float *x, Nd4jLong *xShapeInfo, float *y, Nd4jLong *yShapeInfo, float *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            DISPATCH_SIMPLE(broadcastSimple, float, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

	        DEBUG_KERNEL(stream, opNum);
        }

        template <>
        __host__ void Broadcast<float16>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, float16 *x, Nd4jLong *xShapeInfo, float16 *y, Nd4jLong *yShapeInfo, float16 *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            DISPATCH_SIMPLE(broadcastSimple, float16, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        __host__ void Broadcast<double>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, double *x, Nd4jLong *xShapeInfo, double *y, Nd4jLong *yShapeInfo, double *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            DISPATCH_SIMPLE(broadcastSimple, double, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

            DEBUG_KERNEL(stream, opNum);
        }


        template <typename T>
        template <typename OpType>
		__device__ void Broadcast<T>::transformCuda(
		T *x,
		Nd4jLong *xShapeInfo,
		T *y,
		Nd4jLong *yShapeInfo,
		T *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {

		//decompose in to several sub tads after
		//moving all dimensions (in sorted order)
		//to the back.
		//permuted version of the x shape info for setting up the tad problem
	  __shared__ Nd4jLong tadLength;
      __shared__ Nd4jLong tadEWS;
      __shared__ int tadRank;
      __shared__ int numTads;
      __shared__ Nd4jLong *tadShape;
      __shared__ Nd4jLong *tadStride;
      __shared__ Nd4jLong yEWS;
      __shared__ Nd4jLong zEWS;
      __shared__ int zRank;
      __shared__ Nd4jLong *zShape;
      __shared__ Nd4jLong *zStride;
      __shared__ int yRank;
      __shared__ Nd4jLong *yShape;
      __shared__ Nd4jLong *yStride;
      if (threadIdx.x == 0) {
        if (tadOnlyShapeInfoZ == nullptr) {
            tadOnlyShapeInfoZ = tadOnlyShapeInfo;
            tadOffsetsZ = tadOffsets;
        }

   	    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        yEWS = shape::elementWiseStride(yShapeInfo);
      	zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);

        if (tadEWS < 1 || zEWS < 1 || yEWS < 1 || dimensionLength > 1) {
            tadRank = shape::rank(tadOnlyShapeInfo);
            tadShape = shape::shapeOf(tadOnlyShapeInfo);
      	    tadStride = shape::stride(tadOnlyShapeInfo);
      	    zRank = shape::rank(tadOnlyShapeInfoZ);
      	    zShape = shape::shapeOf(tadOnlyShapeInfoZ);
      	    zStride = shape::stride(tadOnlyShapeInfoZ);
      	    yRank = shape::rank(yShapeInfo);
      	    yShape = shape::shapeOf(yShapeInfo);
      	    yStride = shape::stride(yShapeInfo);
        }
      }
      __syncthreads();

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {


            __shared__ Nd4jLong tadOffsetForBlock;
            __shared__ Nd4jLong tadOffsetForBlockZ;
            __shared__ T *rR;
            __shared__ T *rX;
            if (threadIdx.x == 0) {
                tadOffsetForBlockZ = tadOffsetsZ[r];
                if (result != x)
                    tadOffsetForBlock = tadOffsets[r];
                else
                    tadOffsetForBlock = tadOffsetForBlockZ;

                rR = result + tadOffsetForBlockZ;
                rX = x + tadOffsetForBlock;
            }
            __syncthreads();


            if(tadEWS > 0 && zEWS > 0 && yEWS > 0 && dimensionLength == 1) {
            	if (tadEWS == 1 && yEWS == 1 && zEWS == 1) {
                	for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i] = OpType::op(rX[i], y[i]);
                	}
                } else {
					for (int i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    	rR[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yEWS]);
                	}
                }
            }
            else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong zCoord[MAX_RANK];

                for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {

                    if (shape::order(tadOnlyShapeInfo) == 'c') {
                        shape::ind2subC(tadRank,tadShape, i, tadLength, xCoord);
                        shape::ind2subC(yRank, yShape, i, tadLength, yCoord);
                    } else {
                        shape::ind2sub(tadRank,tadShape, i, tadLength, xCoord);
                        shape::ind2sub(yRank, yShape, i, tadLength, yCoord);
                    }

                    if (shape::order(tadOnlyShapeInfoZ) == 'c')
                        shape::ind2subC(zRank,zShape, i, tadLength, zCoord);
                    else
                        shape::ind2sub(zRank,zShape, i, tadLength, zCoord);

                    auto xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                    auto zOffset = shape::getOffset(tadOffsetForBlockZ, zShape, zStride, zCoord, zRank);
                    auto yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                    result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            }
		}
	}
    }
}