#pragma once
#include <ops.h>

namespace functions {
	namespace broadcast {
		template <typename T>
		class Broadcast;
	}

	namespace transform {
		template <typename T>
		class Transform;
	}

	namespace reduce {
		template <typename T>
		class ReduceFunction;
	}
}

namespace simdOps {

	template<typename T>
	class Im2col {
	public:
		static const bool requiresSpecial = true;
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)

#endif
		static int outSize(int size, int k, int s, int p, bool coverAll) {
			if (coverAll)
				return (size + p * 2 - k + s - 1) / s + 1;
			else
				return (size + p * 2 - k) / s + 1;
		}

#ifdef __CUDACC__
		/**
		* Based on:  https://github.com/pjreddie/darknet/blob/master/src/im2col_kernels.cu
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
			int kernelWidth = (int)extraParams[0];
			int kernelHeight = (int)extraParams[1];
			int strideX = (int)extraParams[2];
			int strideY = (int)extraParams[3];
			int padWidth = (int)extraParams[4];
			int padHeight = (int)extraParams[5];
			int kSize = kernelWidth * kernelHeight;

			int *outShape = shape::shapeOf(resultShapeBuffer);
			char resultOrder = shape::order(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);

			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);

			int samples = inShape[0];
			int depth = inShape[1];
			int height = inShape[2];
			int width = inShape[3];


			int strideex = inStride[0];
			int stridech = inStride[1];
			int strideh = inStride[2];
			int stridew = inStride[3];

			// (height + 2 * padHeight - kernelHeight) / strideX + 1; //
			// (width + 2 * padWidth - kernelWidth) / strideY + 1; //
			int height_col = outShape[4];
			int width_col = outShape[5];

			int n = samples * depth * height_col * width_col;
			/*
			if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, height, width, depth, n, samples);
			*/

			int index = blockIdx.x * blockDim.x + threadIdx.x;
			for (; index < n; index += blockDim.x*gridDim.x) {
				int h_index = index / width_col;
				int h_col = h_index % height_col;
				int w_col = index % width_col;

				int c_im = h_index / height_col;
				int c_col = c_im * kSize;

				int depth_im = c_im % depth;
				int num_im = c_im / depth;
				int h_offset = h_col * strideY - padHeight;
				int w_offset = w_col * strideX - padWidth;

				T* data_col_ptr = result;

				int i_c = (c_col * height_col + h_col) * width_col + w_col;
				data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

				T* data_im_ptr = dx;

				data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

				for (int i = 0; i < kernelHeight; ++i) {
					for (int j = 0; j < kernelWidth; ++j) {
						int h_im = h_offset + i;
						int w_im = w_offset + j;
						int i_f = 0;
						int i_c_temp = i_c;
						for (int dim = 5; dim >= 0; dim--)
						{
							i_f += (i_c_temp % outShape[dim])  * outStride[dim];
							i_c_temp = i_c_temp / outShape[dim];
						}
						if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) result[i_f] = data_im_ptr[i * strideh + j*stridew];
							else result[i_f] = 0;

						//result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
						data_col_ptr += height_col * width_col;
						i_c += height_col * width_col;
					}
				}
			}
		}
#endif


		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *tadShapeInfo, int *tadOffsets) {
			/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
			int kernelWidth = (int)extraParams[0];
			int kernelHeight = (int)extraParams[1];
			int strideX = (int)extraParams[2];
			int strideY = (int)extraParams[3];
			int padWidth = (int)extraParams[4];
			int padHeight = (int)extraParams[5];
			bool coverAll = extraParams[6] > 0.0;

			int outArrayOffset = 0;
			int *outShape = shape::shapeOf(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);

			int inArrayOffset = 0;
			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);


			int exampleFrom = 0;
			int exampleTo = inShape[0];
			int depthFrom = 0;
			int depthTo = inShape[1];
			int yOutFrom = 0;
			int yOutTo = outSize(inShape[2], kernelHeight, strideY, padHeight, coverAll);
			int xOutFrom = 0;
			int xOutTo = outSize(inShape[3], kernelWidth, strideX, padWidth, coverAll);

			T *dIn = dx;
			T *dOut = result;

			int tadsPerThread = (exampleTo - exampleFrom) / 4;
			int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
			num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for num_threads(num_threads) if (num_threads > 1) collapse(2) proc_bind(AFFINITY) default(shared)
			for (int ex = exampleFrom; ex < exampleTo; ex++) {
				for (int d = depthFrom; d < depthTo; d++) {
					int outIndices[6];
					int inIndices[4];

					int inStride2 = inStride[2];
					int inStride3 = inStride[3];
					int outStride2 = outStride[2];
					int outStride3 = outStride[3];
					int inShape2 = inShape[2];
					int inShape3 = inShape[3];

					bool padding = padHeight > 0 || padWidth > 0;
					inIndices[0] = ex;
					inIndices[1] = d;
					outIndices[0] = ex;
					outIndices[1] = d;

					for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
						for (int y = yOutFrom; y < yOutTo; y++) {  //along height
							outIndices[4] = y;
							outIndices[5] = x;
							int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride,
								outIndices);

							if (padding) {
								int i = y * strideY -
									padHeight;    //index along height of first element of patch in original img
								int j = x * strideX -
									padWidth;     //index along width of first element in patch in original img
								inIndices[2] = i;   //along height
								inIndices[3] = j;   //along width

								int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride,
									inIndices);
								if (outStride2 <= outStride3) {
									//Want dimension 2 (along height) in inner loop for cache reasons
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										int outBufferIdxX = baseOffsetOut + patchX * outStride3;
										int inBufferIdxX = baseOffsetIn + patchX * inStride3;
										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 ||
												j + patchX >= inShape3)
												dOut[outBufferIdxX + patchY * outStride2] = 0; //padding
											else {
												dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX +
													patchY *
													inStride2];
											}
										}
									}
								}
								else {
									//Want dimension 3 in inner loop for cache reasons
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										int outBufferIdxY = baseOffsetOut + patchY * outStride2;
										int inBufferIdxY = baseOffsetIn + patchY * inStride2;
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] ||
												j + patchX >= inShape[3])
												dOut[outBufferIdxY + patchX * outStride3] = 0.0; //padding
											else {
												dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY +
													patchX *
													inStride3];
											}
										}
									}
								}
							}
							else {
								//No padding
								int i = y *
									strideY;    //index along height of first element of patch in original img
								int j = x *
									strideX;     //index along width of first element in patch in original img
								inIndices[2] = i;   //along height
								inIndices[3] = j;   //along width

								int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride,
									inIndices);
								if (outStride2 <= outStride3) {
									//Want dimension 2 (along height) in inner loop for cache reasons
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										int outBufferIdxX = baseOffsetOut + patchX * outStride3;
										int inBufferIdxX = baseOffsetIn + patchX * inStride3;
										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX +
												patchY * inStride2];
										}
									}
								}
								else {
									//Want dimension 3 in inner loop for cache reasons
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										int outBufferIdxY = baseOffsetOut + patchY * outStride2;
										int inBufferIdxY = baseOffsetIn + patchY * inStride2;
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY +
												patchX * inStride3];
										}
									}
								}
							}
						}
					}
				}
			}

		}

		op_def static T op(T d1, T *params) {
			return d1;
		}


		/** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
		*  normally negative indices are bad, OK here because of other checks on input indices
		*  Uses unrolled loop specifically for length 4
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[2] != 1) offset += indices[2] * stride[2];
			if (shape[3] != 1) offset += indices[3] * stride[3];
			return offset;
		}


		/**
		* A version of Shape.getOffset without checking on input for negative indices etc
		* normally negative indices are bad, OK here because of other checks on input indices
		* Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};

	template<typename T>
	class Histogram {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {

            int numBins = (int) extraParams[0];
            T min_val = extraParams[1];
            T max_val = extraParams[2];

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ T *bins;
            __shared__ int length;
            __shared__ T *reductor;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                bins = (T *) shmem;
                reductor = ((T *) allocationPointer) + (numBins * blockIdx.x);

                length = shape::length(xShapeBuffer);
            }
            __syncthreads();

            T binSize = (max_val - min_val) / (numBins);

            for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                bins[e] = (T) 0.0f;
            }
            __syncthreads();

            for (int e = tid; e < length; e+= blockDim.x * gridDim.x) {
                int idx = (int) ((dx[e] - min_val) / binSize);
				    if (idx < 0) idx = 0;
					else if (idx >= numBins) idx = numBins - 1;

				nd4j::math::atomics::nd4j_atomicAdd(&bins[idx], (T) 1.0f);
            }
            __syncthreads();

            // transfer shared memory to reduction memory


            if (gridDim.x > 1) {
                unsigned int *tc = (unsigned int *)reductionPointer;
                __shared__ bool amLast;

                for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                    reductor[e] = bins[e];
                }
                __threadfence();
                __syncthreads();

                if (threadIdx.x == 0) {
						unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
						amLast = (ticket == gridDim.x - 1);
				}
				__syncthreads();

				if (amLast) {
				    tc[16384] = 0;

                    // nullify shared memory for future accumulation
                    for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                        bins[e] = (T) 0.0f;
                    }

                    // accumulate reduced bins
                    for (int r = 0; r < gridDim.x; r++) {
                        T *ptrBuf = ((T *)allocationPointer) + (r * numBins);

                        for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                            bins[e] += ptrBuf[e];
                        }
                    }
                    __syncthreads();

                    // write them out to Z
                    for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                        result[e] = bins[e];
                    }
				}
            } else {
                // if there's only 1 block - just write away data
                for (int e = threadIdx.x; e < numBins; e += blockDim.x) {
                    result[e] = bins[e];
                }
            }

		};
#endif

		static void execSpecial(
				T *dx,
				int *xShapeBuffer,
				T *result,
				int *resultShapeBuffer,
				T *extraParams, int *tadShapeInfo, int *tadOffsets) {

			int length = shape::length(xShapeBuffer);
			int _threads = 2;

			int numBins = (int) extraParams[0];
			int span = (length / _threads) + 8;

			// get min over input
            T min_val = extraParams[1];
            T max_val = extraParams[2];

            /*
#pragma omp parallel for simd num_threads(_threads) if (_threads > 1) reduction(min:min_val) proc_bind(close)
            for (int x = 0; x < length; x++) {
				if (min_val > dx[x])
					min_val = dx[x];
			}

			// get max over input
			T max_val = (T) MIN_FLOAT;

#pragma omp parallel for simd num_threads(_threads) if (_threads > 1) reduction(max:max_val) proc_bind(close)
			for (int x = 0; x < length; x++) {
				if (max_val < dx[x])
					max_val = dx[x];
			}
            */

			T binSize = (max_val - min_val) / (numBins);


#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(close) default(shared)
			{
				int tid, start, end;

				int *bins = new int[numBins];
                std::memset(bins, 0, sizeof(int) * numBins);
				tid = omp_get_thread_num();
				start = span * tid;
				end = span * (tid + 1);
				if (end > length) end = length;

#pragma omp simd
				for (int x = start; x < end; x++) {
					int idx = (int) ((dx[x] - min_val) / binSize);
					if (idx < 0)
						idx = 0;
					else if (idx >= numBins)
						idx = numBins - 1;

					bins[idx]++;
				}

#pragma omp critical
				{
#pragma omp simd
					for (int x = 0; x < numBins; x++) {
						result[x] += bins[x];
					}

				}

				delete[] bins;
			}

		}


        op_def static T op(T d1, T *params) {
            return d1;
        }
	};

	template<typename T>
	class Col2Im {

	public:
		static const bool requiresSpecial = true;
#ifdef __CUDACC__
		/**
		* https://github.com/pjreddie/darknet/blob/master/src/col2im_kernels.cu
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);

			int strideex = inStride[0];
			int stridech = inStride[1];
			int stridekrow = inStride[2];
			int stridekcol = inStride[3];
			int striderow = inStride[4];
			int stridecol = inStride[5];

			int kernelHeight = inShape[2];
			int kernelWidth = inShape[3];

			// C

			int strideX = (int)extraParams[0];
			int strideY = (int)extraParams[1];
			int padWidth = (int)extraParams[2];
			int padHeight = (int)extraParams[3];
			int imgHeight = (int)extraParams[4];
			int imgWidth = (int)extraParams[5];

			int *outShape = shape::shapeOf(resultShapeBuffer);
			char resultOrder = shape::order(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);

			int samples = outShape[0];
			int depth = outShape[1];
			//int height = outShape[2];
			//int width = outShape[3];

			int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
			int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

			int n = samples * depth * imgHeight * imgWidth;

			/*if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, imgHeight, imgWidth, depth, n, samples);*/



			for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
				T val = 0;
				int w_im = i % imgWidth + padWidth;
				int h_im = (i / imgWidth) % imgHeight + padHeight;
				int c_im = i / (imgWidth * imgWidth);

				int num_im = c_im / depth;
				int depth_im = c_im % depth;

				// compute the start and end of the output
				int w_col_start = (w_im < kernelWidth) ? 0 : (w_im - kernelWidth) / strideX + 1;
				int w_col_end = nd4j::math::nd4j_min<int>(w_im / strideX + 1, width_col);

				int h_col_start = (h_im < kernelHeight) ? 0 : (h_im - kernelHeight) / strideY + 1;
				int h_col_end = nd4j::math::nd4j_min<int>(h_im / strideY + 1, height_col);


				for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
					for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
						int h_k = (h_im - h_col * strideY);
						int w_k = (w_im - w_col * strideX);

						int data_col_index = num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;

						val += dx[data_col_index];
					}
				}
				int i_f = 0;
				int i_c = i;
				for (int dim = 3; dim >= 0; dim--)
				{
					i_f += (i_c % outShape[dim])  * outStride[dim];
					i_c = i_c / outShape[dim];
				}
				result[i_f] += val;
			}
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *tadShapeInfo, int *tadOffsets) {
			int inOffset = 0;
			int *inShape = shape::shapeOf(xShapeBuffer);
			int *inStride = shape::stride(xShapeBuffer);

			int kernelHeight = inShape[2];
			int kernelWidth = inShape[3];
			/* int strideY, int strideX, int padHeight, int padWidth, int imgHeight, int imgWidth, */
			int strideX = (int)extraParams[0];
			int strideY = (int)extraParams[1];
			int padWidth = (int)extraParams[2];
			int padHeight = (int)extraParams[3];


			int exampleFrom = 0;
			int exampleTo = inShape[0];
			int depthFrom = 0;
			int depthTo = inShape[1];

			int outArrayOffset = 0;
			int *outShape = shape::shapeOf(resultShapeBuffer);
			int *outStride = shape::stride(resultShapeBuffer);


			int tadsPerThread = (exampleTo - exampleFrom) / 4;
			int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
			num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

			T *fIn = dx;
			T *fOut = result;
#pragma omp parallel for num_threads(num_threads) if (num_threads>1) collapse(2) proc_bind(AFFINITY) default(shared)
			for (int ex = exampleFrom; ex < exampleTo; ex++) {
				for (int d = depthFrom; d < depthTo; d++) {
					int outIndices[4];
					int inIndices[6];

					int inStride2 = inStride[2];
					int inStride3 = inStride[3];
					int outStride2 = outStride[2];
					int outStride3 = outStride[3];
					int outShape2 = outShape[2];
					int outShape3 = outShape[3];

					int yOutTo = inShape[4];
					int xOutTo = inShape[5];


					bool padding = padHeight > 0 || padWidth > 0;
					inIndices[0] = ex;
					inIndices[1] = d;
					outIndices[0] = ex;
					outIndices[1] = d;

					for (int x = 0; x < xOutTo; x++) {  //Patch number along width
						for (int y = 0; y < yOutTo; y++) {  //Patch number along height
							inIndices[4] = y;   //patch number (along height)
							inIndices[5] = x;   //patch number (along width)
							int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

							if (padding) {
								int i = y * strideY -
									padHeight;    //index along height of first element of patch in original img
								int j = x * strideX -
									padWidth;     //index along width of first element in patch in original img
								outIndices[2] = i;  //along height
								outIndices[3] = j;  //along width

								int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride,
									outIndices);

								if (inStride2 <= inStride3) {
									//Want dimension 2 (along height) in inner loop for cache efficiency
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										if (j + patchX < 0 || j + patchX >= outShape3)
											continue;

										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											if (i + patchY < 0 || i + patchY >= outShape2)
												continue;
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
								else {
									//Want dimension 3 (along width) in inner loop for cache efficiency
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										if (i + patchY < 0 || i + patchY >= outShape2)
											continue;
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											if (j + patchX < 0 || j + patchX >= outShape3)
												continue;
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
							}
							else {
								//No padding
								int i = y *
									strideY;    //index along height of first element of patch in output img
								int j = x *
									strideX;     //index along width of first element in patch in output img

								outIndices[2] = i;
								outIndices[3] = j;

								int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride,
									outIndices);

								if (inStride2 <= inStride3) {
									//Want dimension 2 (along height) in inner loop for cache efficiency
									for (int patchX = 0; patchX < kernelWidth; patchX++) {
										for (int patchY = 0; patchY < kernelHeight; patchY++) {
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
								else {
									//Want dimension 3 (along width) in inner loop for cache efficiency
									for (int patchY = 0; patchY < kernelHeight; patchY++) {
										for (int patchX = 0; patchX < kernelWidth; patchX++) {
											fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
												fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
										}
									}
								}
							}
						}
					}
				}
			}


		}

		op_def static T op(T d1, T *params) {
			return d1;
		}


		/** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
		*  normally negative indices are bad, OK here because of other checks on input indices
		*  Uses unrolled loop specifically for length 4
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[2] != 1) offset += indices[2] * stride[2];
			if (shape[3] != 1) offset += indices[3] * stride[3];
			return offset;
		}

		/** A version of Shape.getOffset without checking on input for negative indices etc
		* normally negative indices are bad, OK here because of other checks on input indices
		* Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
		*/
#ifdef __CUDACC__
		inline __host__ __device__
#elif defined(__GNUC__)


#endif
		static int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};



	template<typename T>
	class SoftMax {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {

			int *shape = shape::shapeOf(xShapeBuffer);
			__shared__ T maxResult;
			__shared__ int *maxResultShapeBuffer;

			int length = shape::length(xShapeBuffer);

			if (threadIdx.x == 0) {
				maxResult = (T) 0.0;
			}
			__syncthreads();

			int *stride = shape::stride(xShapeBuffer);
			//compute the row wise maxes

			int maxShape[2] = { shape[0], 1 };

			// it's always 2d here
			__shared__ int tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);

			functions::reduce::ReduceFunction<T>::template execScalarCuda<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Subtract<T>>(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			functions::transform::Transform<T>::template transformCuda<simdOps::Exp<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
			__syncthreads();

			//take the sum for the exponential
			functions::reduce::ReduceFunction<T>::template execScalarCuda<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Divide<T>>(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *tadShapeInfo, int *tadOffsets) {
			if (shape::isMatrix(xShapeBuffer)) {
				int *shape = shape::shapeOf(xShapeBuffer);
				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				//compute the row wise maxes
				std::vector <T> maxResult(shape[0]);
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;
				int maxShape[2] = { shape[0], 1 };
				int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<T>::template exec<simdOps::Subtract<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr, nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<T>::template exec<simdOps::Exp<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr, nullptr, nullptr);

				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer)) {
				T max = 0;
				T sum = 0;
				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride >= 1 && resultElementWiseStride >= 1) {
					if (elementWiseStride == 1 && resultElementWiseStride == 1) {

#pragma omp simd reduction(max:max)
						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<T>(max, dx[i]);
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i] = dx[i] - max;
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							sum += result[i];
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i] /= sum;
						}
					}
					else {

#pragma omp simd reduction(max:max)
						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<T>(max, dx[i * elementWiseStride]);
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] = dx[i * elementWiseStride] - max;
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * resultElementWiseStride]);
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							sum += result[i * resultElementWiseStride];
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] /= sum;
						}
					}
				}
			}
		}

		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};



	template<typename T>
	class LogSoftMax {
	public:
		static const bool requiresSpecial = true;
#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			int *shape = shape::shapeOf(xShapeBuffer);
			int *stride = shape::stride(xShapeBuffer);
			//iterate along rows

			__shared__ T maxResult;
			__shared__ int *maxResultShapeBuffer;
			if (threadIdx.x == 0) {

				maxResult = (T) 0.0;
			}
			__syncthreads();
			//compute the row wise maxes

			int maxShape[2] = { shape[0], 1 };
			__shared__ int tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);

			functions::reduce::ReduceFunction<T>::template execScalarCuda<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Subtract<T>>(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			functions::transform::Transform<T>::template transformCuda<simdOps::Exp<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
			__syncthreads();

			//take the sum for the exponential
			functions::reduce::ReduceFunction<T>::template execScalarCuda<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Divide<T>>(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			functions::transform::Transform<T>::template transformCuda<simdOps::Log<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
		}
#endif


		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *tadShapeInfo, int *tadOffsets) {

			if (shape::isMatrix(xShapeBuffer, 2)) {
				int *shape = shape::shapeOf(xShapeBuffer);
				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				//compute the row wise maxes
				std::vector <T> maxResult(shape[0]);

#pragma omp simd
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;

				int maxShape[2] = { shape[0], 1 };
				int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<T>::template exec<simdOps::Subtract<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr, nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<T>::template exec<simdOps::Exp<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr, nullptr, nullptr);

				functions::transform::Transform<T>::template exec<simdOps::Log<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);


				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				T max = 0;
				T sum = 0;

				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {
#pragma omp simd reduction(max:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i]);
					}

#pragma omp simd reduction(+:sum)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						sum += result[i];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
						result[i] = nd4j::math::nd4j_log<T>(result[i]);
					}
				}
				else {
#pragma omp simd reduction(max:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
					}

#pragma omp simd reduction(+:sum)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
						result[i * elementWiseStride] = nd4j::math::nd4j_log<T>(result[i * elementWiseStride]);
					}
				}
			}
		}

		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};


	/**
	* softmax(x)
	*/
	template<typename T>
	class SoftMaxDerivative {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {


			int *shape = shape::shapeOf(xShapeBuffer);
			__shared__ T maxResult;
			__shared__ int *maxResultShapeBuffer;
			__shared__ int resultEWS;

			int length = shape::length(xShapeBuffer);

			if (threadIdx.x == 0) {
				resultEWS = shape::elementWiseStride(resultShapeBuffer);

				maxResult = (T) 0.0;
			}
			__syncthreads();

			int *stride = shape::stride(xShapeBuffer);
			int maxShape[2] = { shape[0], 1 };

			__shared__ int tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);

			functions::reduce::ReduceFunction<T>::template execScalarCuda<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Subtract<T>>(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			functions::transform::Transform<T>::template transformCuda<simdOps::Exp<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager);
			__syncthreads();

			//take the sum for the exponential
			functions::reduce::ReduceFunction<T>::template execScalarCuda<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Divide<T>>(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			if (resultEWS >= 1) {
				for (int i = threadIdx.x; i < length; i += blockDim.x) {
					result[i * resultEWS] = result[i * resultEWS] * ((T) 1.0 - result[i * resultEWS]);
				}
			}
			else {
				printf("Non element wise stride not supported right now\n");
			}
		}
#endif


		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *tadShapeInfo, int *tadOffsets) {
			if (shape::isMatrix(xShapeBuffer, 2)) {
				int *shape = shape::shapeOf(xShapeBuffer);

				int resultEleStide = shape::elementWiseStride(resultShapeBuffer);

				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				int len = shape::length(xShapeBuffer);
				//compute the row wise maxes
				std::vector <T> maxResult(shape[0]);
#pragma omp simd
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;

				int maxShape[2] = { shape[0], 1 };
				int *maxResultShapeBuffer = shape::shapeBuffer(2, maxShape);
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<T>::template exec<simdOps::Subtract<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr, nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<T>::template exec<simdOps::Exp<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension,
					1, nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1, nullptr, nullptr,
                                                                                      nullptr, nullptr);

				if (resultEleStide >= 1) {
					if (resultEleStide == 1) {
#pragma omp simd
						for (int i = 0; i < len; i++) {
							result[i] = result[i] * (1 - result[i]);
						}

					}
					else {
#pragma omp simd
						for (int i = 0; i < len; i++) {
							result[i * resultEleStide] = result[i * resultEleStide] * (1 - result[i * resultEleStide]);
						}

					}
				}

				else {
					printf("Non element wise stride not supported right now\n");
				}


				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				T max = 0;
				T sum = 0;

				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {

#pragma omp simd reduction(max:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i]);
					}

#pragma omp simd reduction(+:sum)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						sum += result[i];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
					}

#pragma omp simd
                    for (int i = 0; i < length; i++) {
                        result[i] = result[i] * (1 - result[i]);
                    }
                } else {

#pragma omp simd reduction(max:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
					}


#pragma omp simd reduction(+:sum)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] = result[i * elementWiseStride] * (1 - result[i * elementWiseStride]);
					}
				}
			}
		}

		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};


	template<typename T>
	class IsMax {
	public:
		static const bool requiresSpecial = true;


#ifdef __CUDACC__

		static inline  __device__ void doAllCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {

			__shared__ int maxIdx;
			__shared__ int length;
			if (threadIdx.x == 0) {
				length = shape::length(resultShapeBuffer);
			}
			__syncthreads();

			functions::indexreduce::IndexReduce<T>::template transform<simdOps::IndexMax<T>>(
				dx,
				xShapeBuffer,
				extraParams,
				result,
				resultShapeBuffer,
				nullptr,
				1,
				1, allocationPointer, reductionPointer, manager, nullptr, nullptr);

			__syncthreads();
			if (threadIdx.x == 0)
				maxIdx = (int)result[0];
			__syncthreads();

			for (int i = threadIdx.x; i < length; i += blockDim.x)
				result[i] = 0;
			__syncthreads();

			if (threadIdx.x == 0) {
				result[maxIdx] = 1.0;
			}

		}
#endif

#ifdef __CUDACC__
		inline __host__

#elif defined(__GNUC__)


#endif
		static void doAll(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {

			int length = shape::length(xShapeBuffer);
			int eleStride = shape::elementWiseStride(xShapeBuffer);
			int resultEleStride = shape::elementWiseStride(resultShapeBuffer);
			char xOrder = shape::order(xShapeBuffer);
			char resultOrder = shape::order(resultShapeBuffer);
/*
			int tadsPerThread = tads / TAD_THRESHOLD;
			int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
			num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());
*/
			if (xOrder == resultOrder && xOrder == 'c') {
				if (eleStride == 1 && resultEleStride == 1) {
					if (length < ELEMENT_THRESHOLD) {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp simd reduction (max:maxIdx,currMax)
						for (int i = 0; i < length; i++) {
							if (currMax < dx[i]) {
								currMax = dx[i];
								maxIdx = i;
							}

							result[i] = 0.0;

						}

						result[maxIdx] = 1.0;

					}
					else {
						int maxIdx = 0;
						T currMax = dx[0];

#pragma omp parallel proc_bind(AFFINITY)
{
						int maxIdxLocal = maxIdx;
						T currMaxLocal = currMax;

#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
						for (int i = 0; i < length; i++) {
							if (currMaxLocal < dx[i]) {
								currMaxLocal = dx[i];
								maxIdxLocal = i;
							}
							result[i] = 0.0;
						}
#pragma omp critical
{
						if (currMax < currMaxLocal) {
							currMax = currMaxLocal;
							maxIdx = maxIdxLocal;
						}
}
}
						result[maxIdx] = 1.0;
					}

				}
				else {
					if (length < ELEMENT_THRESHOLD) {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp simd reduction(max:maxIdx,currMax)
						for (int i = 0; i < length; i++) {
							result[i * resultEleStride] = 0.0;
							if (currMax < dx[i * eleStride]) {
								currMax = dx[i * eleStride];
								maxIdx = i;
							}
						}

						result[maxIdx * resultEleStride] = 1.0;

					}
					else {
						int maxIdx = 0;
						T currMax = dx[0];

#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
						int maxIdxLocal = maxIdx;
						T currMaxLocal = currMax;
#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
						for (int i = 0; i < length; i++) {
							result[i * resultEleStride] = 0.0;
							if (currMaxLocal < dx[i * eleStride]) {
								currMaxLocal = dx[i * eleStride];
								maxIdxLocal = i;
							}
						}

#pragma omp critical
{
						if (currMax < currMaxLocal) {
							currMax = currMaxLocal;
							maxIdx = maxIdxLocal;
						}
}
}
						result[maxIdx * resultEleStride] = 1.0;
					}

				}
			}


			else {
				int shapeIter[MAX_RANK];
				int coord[MAX_RANK];
				int dim;
				int xStridesIter[MAX_RANK];
				int resultStridesIter[MAX_RANK];
				int *xShape = shape::shapeOf(xShapeBuffer);
				int *xStride = shape::stride(xShapeBuffer);
				int *resultStride = shape::stride(resultShapeBuffer);
				int rank = shape::rank(xShapeBuffer);
				T *originalResult = result;
				if (PrepareTwoRawArrayIter<T>(rank,
					xShape,
					dx,
					xStride,
					result,
					resultStride,
					&rank,
					shapeIter,
					&dx,
					xStridesIter,
					&result,
					resultStridesIter) >= 0) {
					T value = dx[0];
					int idx = 0;
					int maxIdx = 0;
					ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
						if (dx[0] > value) {
							value = dx[0];
							maxIdx = idx;
						}

						idx++;
						result[0] = 0.0;

					}
					ND4J_RAW_ITER_TWO_NEXT(
						dim,
						rank,
						coord,
						shapeIter,
						dx,
						xStridesIter,
						result,
						resultStridesIter);

					//pointer to where max value would be
					if (shape::order(resultShapeBuffer) == 'c' || (shape::order(resultShapeBuffer) == 'f' &&
						maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1] >=
						shape::length(resultShapeBuffer)))
						originalResult[maxIdx] = 1.0;
					else
						originalResult[maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1]] = 1.0;
				}
			}


		}
	public:


#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
			// FIXME: MAX_DIMENSION is lower then FP16 frame
			if (extraParams == nullptr || (int) extraParams[0] == MAX_DIMENSION) {
				doAllCuda(dx, xShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, reductionPointer, manager);
			}
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *tadShapeInfo, int *tadOffsets) {

			if (extraParams == nullptr || extraParams[0] == 0 ||
				(extraParams[0] == 1 && extraParams[1] == MAX_DIMENSION)) {
				doAll(dx, xShapeBuffer, result, resultShapeBuffer, extraParams);
			}
			else if (shape::isVector(xShapeBuffer)) {
				int dimensionLength = (int)extraParams[0];
				int *dimension = new int[dimensionLength];
				int length = shape::length(xShapeBuffer);
				for (int i = 0; i < dimensionLength; i++) {
					dimension[i] = (int)extraParams[i + 1];
				}
				if (shape::shapeOf(xShapeBuffer)[dimension[0]] == 1) {
					for (int i = 0; i < length; i++) {
						result[i] = 1.0;
					}
				}
				else {
					int eleStride = shape::elementWiseStride(xShapeBuffer);
					if (eleStride == 1) {
						int maxIdx = 0;
						T currMax = dx[0];
						if (length < ELEMENT_THRESHOLD) {

#pragma omp simd reduction(max:maxIdx,currMax)
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i]) {
									currMax = dx[i];
									maxIdx = i;
								}

								dx[i] = 0.0;

							}
						}
						else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
							int maxIdxLocal = maxIdx;
							T currMaxLocal = currMax;
#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
							for (int i = 0; i < length; i++) {
								if (currMaxLocal < dx[i]) {
									currMaxLocal = dx[i];
									maxIdxLocal = i;
								}

								result[i] = 0.0;

							}
#pragma omp critical
                            {
							    if (currMax < currMaxLocal) {
								    currMax = currMaxLocal;
								    maxIdx = maxIdxLocal;
							    }
                            }
}
						}

						result[maxIdx] = 1.0;

					}


					else {
						int maxIdx = 0;
						T currMax = dx[0];
						if (length < ELEMENT_THRESHOLD) {
#pragma omp parallel for reduction(max:maxIdx,currMax) proc_bind(AFFINITY)
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i * eleStride]) {
									currMax = dx[i * eleStride];
									maxIdx = i;
								}

								result[i] = 0.0;
							}
						}
						else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
							int maxIdxLocal = maxIdx;
							T currMaxLocal = currMax;

#pragma omp parallel for reduction(max:maxIdx,currMax)  proc_bind(AFFINITY)
							for (int i = 0; i < length; i++) {
								if (currMaxLocal < dx[i * eleStride]) {
									currMaxLocal = dx[i * eleStride];
									maxIdxLocal = i;
								}

								result[i] = 0.0;
							}
#pragma omp critical
{
							if (currMax < currMaxLocal) {
								currMax = currMaxLocal;
								maxIdx = maxIdxLocal;
							}
}
}
						}

						result[maxIdx] = 1.0;
					}
				}


			}
			else {
                int dimensionLength = (int) extraParams[0];
                int *dimension = new int[dimensionLength];

#pragma omp simd
                for (int i = 0; i < dimensionLength; i++) {
                    dimension[i] = (int) extraParams[i + 1];
                }

/*
                shape::TAD tad(xShapeBuffer, dimension, dimensionLength);
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();
*/
//                int tads = tad.numTads;
                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                int *tadShapeShapeInfo = tadShapeInfo;

                int tadLength = shape::tadLength(xShapeBuffer, dimension, dimensionLength);
                int tads = shape::length(xShapeBuffer) / tadLength;

                int tadsPerThread = tads / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                int tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                int zEWS = tadEWS;

                int span = (tads / num_threads) + 8;

//#pragma omp parallel for num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY)
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY)
                {
                    int tid = omp_get_thread_num();
                    int start = span * tid;
                    int end = span * (tid + 1);
                    if (end > tads) end = tads;

                    for (int r = start; r < end; r++) {
                        if (tadEWS > 0 && zEWS > 0) {
                            T *rX = dx + tadOffsets[r];
                            T *rZ = result + tadOffsets[r];

                            T maxValue = rX[0];
                            int maxIdx = 0;
                            if (tadEWS == 1 && zEWS == 1) {
#pragma omp simd reduction(max:maxValue,maxIdx)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i] = maxIdx == i ? (T) 1.0 : (T) 0.0;
                                }

                            } else {

#pragma omp parallel for reduction(max:maxValue,maxIdx) default(shared)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i * tadEWS] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i * tadEWS];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i * zEWS] = maxIdx == i ? (T) 1.0 : (T) 0.0;
                                }
                            }
                        } else {
                            int tadsPerThread = tads / TAD_THRESHOLD;
                            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                            for (int i = 0; i < tads; i++) {
                                int offset = tadOffsets[i];
                                int shapeIter[MAX_RANK];
                                int coord[MAX_RANK];
                                int dim;
                                int xStridesIter[MAX_RANK];
                                int resultStridesIter[MAX_RANK];
                                int *xShape = shape::shapeOf(tadShapeShapeInfo);
                                int *xStride = shape::stride(tadShapeShapeInfo);
                                int *resultStride = shape::stride(tadShapeShapeInfo);
                                int rank = shape::rank(tadShapeShapeInfo);
                                T *xPointer = dx + offset;
                                T *resultPointer = result + offset;
                                T maxValue = xPointer[0];

                                T *maxCursor = resultPointer;
                                Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                                if (PrepareTwoRawArrayIter<T>(rank,
                                                              xShape,
                                                              xPointer,
                                                              xStride,
                                                              resultPointer,
                                                              resultStride,
                                                              &rank,
                                                              shapeIter,
                                                              &xPointer,
                                                              xStridesIter,
                                                              &resultPointer,
                                                              resultStridesIter) >= 0) {
                                    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                        if (maxValue < xPointer[0]) {
                                            maxCursor = resultPointer;
                                            maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
                                            maxValue = xPointer[0];
                                        }

                                        resultPointer[0] = 0.0;
                                    }
                                    ND4J_RAW_ITER_TWO_NEXT(dim,
                                                           rank,
                                                           coord,
                                                           shapeIter,
                                                           xPointer,
                                                           xStridesIter,
                                                           resultPointer,
                                                           resultStridesIter);
                                    maxCursor = reinterpret_cast<T *>(maxCursorLong);
                                    maxCursor[0] = 1.0;
                                }
                            }
                        }
                    }
                }

                delete[] dimension;
            }
		}

		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};
}