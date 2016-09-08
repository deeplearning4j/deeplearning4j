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
			T *extraParams) {
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
#pragma omp parallel for collapse(2)
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
			T *extraParams) {
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


			T *fIn = dx;
			T *fOut = result;
#pragma omp parallel for collapse(2)
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
			T *extraParams) {
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
				functions::transform::Transform<T>::template exec<simdOps::Exp<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

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
						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<T>(max, dx[i]);
						}


						for (int i = 0; i < length; i++) {
							result[i] = dx[i] - max;
						}

						for (int i = 0; i < length; i++) {
							result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						}


						for (int i = 0; i < length; i++) {
							sum += result[i];
						}


						for (int i = 0; i < length; i++) {
							result[i] /= sum;
						}


					}
					else {

						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<T>(max, dx[i * elementWiseStride]);
						}
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] = dx[i * elementWiseStride] - max;
						}
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] = nd4j::math::nd4j_exp<T>(
								result[i * resultElementWiseStride]);
						}
						for (int i = 0; i < length; i++) {
							sum += result[i * resultElementWiseStride];
						}
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
			T *extraParams) {

			if (shape::isMatrix(xShapeBuffer, 2)) {
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
				functions::transform::Transform<T>::template exec<simdOps::Exp<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

				//take the sum for the exponential
				functions::reduce::ReduceFunction<T>::template exec<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1,
					nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<T>::template exec<simdOps::Divide<T>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, dimension, 1,
					nullptr, nullptr, nullptr, nullptr);

				functions::transform::Transform<T>::template exec<simdOps::Log<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);



				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				T max = 0;
				T sum = 0;

				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {
#pragma omp parallel for simd reduction(max:max) shared(result)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i]);
					}

#pragma omp parallel for simd reduction(+:sum)  shared(result)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						sum += result[i];
					}

#pragma omp parallel for simd
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
						result[i] = nd4j::math::nd4j_log<T>(result[i]);
					}
				}
				else {
#pragma omp parallel for simd reduction(max:max) shared(result, elementWiseStride)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
					}

#pragma omp parallel for simd reduction(+:sum)  shared(result, elementWiseStride)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp parallel for simd
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
					result[i * resultEWS] = result[i * resultEWS] * (1 - result[i * resultEWS]);
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
			T *extraParams) {
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
				functions::transform::Transform<T>::template exec<simdOps::Exp<T>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams);

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

#pragma omp parallel for simd reduction(max:max) shared(result) schedule(guided)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i]);
					}

#pragma omp parallel for simd reduction(+:sum)  shared(result) schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<T>(result[i]);
						sum += result[i];
					}

#pragma omp parallel for simd schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
					}

				}
				else {

#pragma omp parallel for simd reduction(max:max) shared(result) schedule(guided)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<T>(max, result[i * elementWiseStride]);
					}


#pragma omp parallel for simd reduction(+:sum) shared(result, elementWiseStride) schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<T>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp parallel for simd schedule(guided)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
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
			if (xOrder == resultOrder && xOrder == 'c') {
				if (eleStride == 1 && resultEleStride == 1) {
					if (length < 8000) {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp simd
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
#pragma omp parallel
{
						int maxIdxLocal = maxIdx;
						T currMaxLocal = currMax;
#pragma omp for nowait
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
					if (length < 8000) {
						int maxIdx = 0;
						T currMax = dx[0];
#pragma omp simd
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
#pragma omp parallel
{
						int maxIdxLocal = maxIdx;
						T currMaxLocal = currMax;
#pragma omp for nowait
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
			if (extraParams == nullptr || extraParams[0] == MAX_DIMENSION) {
				doAllCuda(dx, xShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, reductionPointer, manager);
			}
		}
#endif

		static void execSpecial(
			T *dx,
			int *xShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams) {

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
						if (length < 8000) {
#pragma omp simd
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i]) {
									currMax = dx[i];
									maxIdx = i;
								}

								dx[i] = 0.0;

							}
						}
						else {
#pragma omp parallel
{
							int maxIdxLocal = maxIdx;
							T currMaxLocal = currMax;
#pragma omp for nowait
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
						if (length < 8000) {
#pragma omp simd
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i * eleStride]) {
									currMax = dx[i * eleStride];
									maxIdx = i;
								}

								dx[i] = 0.0;

							}
						}
						else {
#pragma omp parallel
{
							int maxIdxLocal = maxIdx;
							T currMaxLocal = currMax;
#pragma omp for nowait
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
				int dimensionLength = (int)extraParams[0];
				int *dimension = new int[dimensionLength];
				for (int i = 0; i < dimensionLength; i++) {
					dimension[i] = (int)extraParams[i + 1];
				}



				shape::TAD tad(xShapeBuffer, dimension, dimensionLength);
				tad.createTadOnlyShapeInfo();
				tad.createOffsets();

				int tads = tad.numTads;
				//decompose in to several sub tads after
				//moving all dimensions (in sorted order)
				//to the back.
				//permuted version of the x shape info for setting up the tad problem
				int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;
#pragma omp  parallel  for
				for (int i = 0; i < tads; i++) {
					int offset = tad.tadOffsets[i];
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

		op_def static T op(T d1, T *params) {
			return nd4j::math::softplus<T>(d1);
		}
	};
}