//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/broadcasting.h>
#include <loops/legacy_ops.h>


template <typename T, typename OpClass>
__device__ void broadcastSimpleGeneric(
		T *x,
		int *xShapeInfo,
		T *y,
		int *yShapeInfo,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {


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
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float, INPUT(float *x, int *xShapeInfo, float *y, int *yShapeInfo, float *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, double, INPUT(double *x, int *xShapeInfo, double *y, int *yShapeInfo, double *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))
DISPATCH_KERNEL_SIMPLE(broadcastSimple_, broadcastSimpleGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *y, int *yShapeInfo, float16 *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))


namespace functions {
    namespace broadcast {

        template <>
        __host__ void Broadcast<float>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, float *x, int *xShapeInfo, float *y, int *yShapeInfo, float *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {
            DISPATCH_SIMPLE(broadcastSimple, float, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

	        DEBUG_KERNEL(stream, opNum);
        }

        template <>
        __host__ void Broadcast<float16>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, float16 *x, int *xShapeInfo, float16 *y, int *yShapeInfo, float16 *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {
            DISPATCH_SIMPLE(broadcastSimple, float16, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

            DEBUG_KERNEL(stream, opNum);
        }

        template <>
        __host__ void Broadcast<double>::executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, double *x, int *xShapeInfo, double *y, int *yShapeInfo, double *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {
            DISPATCH_SIMPLE(broadcastSimple, double, PARAMS(x, xShapeInfo, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_OPS))

            DEBUG_KERNEL(stream, opNum);
        }


        template <typename T>
        template <typename OpType>
		__device__ void Broadcast<T>::transformCuda(
		T *x,
		int *xShapeInfo,
		T *y,
		int *yShapeInfo,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ) {

		//decompose in to several sub tads after
		//moving all dimensions (in sorted order)
		//to the back.
		//permuted version of the x shape info for setting up the tad problem
	  __shared__ int tadLength;
      __shared__ int tadEWS;
      __shared__ int tadRank;
      __shared__ int numTads;
      __shared__ int *tadShape;
      __shared__ int *tadStride;
      __shared__ int yEWS;
      __shared__ int zEWS;
      __shared__ int zRank;
      __shared__ int *zShape;
      __shared__ int *zStride;
      __shared__ int yRank;
      __shared__ int *yShape;
      __shared__ int *yStride;
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


            __shared__ Nd4jIndex tadOffsetForBlock;
            __shared__ Nd4jIndex tadOffsetForBlockZ;
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
                int xCoord[MAX_RANK];
                int yCoord[MAX_RANK];
                int zCoord[MAX_RANK];

                for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {

                    if (shape::order(tadOnlyShapeInfo) == 'c') {
                        shape::ind2subC(tadRank,tadShape, i, xCoord);
                        shape::ind2subC(yRank, yShape, i, yCoord);
                    } else {
                        shape::ind2sub(tadRank,tadShape, i, xCoord);
                        shape::ind2sub(yRank, yShape, i, yCoord);
                    }

                    if (shape::order(tadOnlyShapeInfoZ) == 'c')
                        shape::ind2subC(zRank,zShape, i, zCoord);
                    else
                        shape::ind2sub(zRank,zShape, i, zCoord);

                    Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);
                    Nd4jIndex zOffset = shape::getOffset(tadOffsetForBlockZ, zShape, zStride, zCoord, zRank);
                    Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                    result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            }
		}
	}
    }
}