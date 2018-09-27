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

#pragma once
#include <ops/ops.h>
#include <loops/reduce_float.h>
#include <loops/scalar.h>
#include <loops/indexreduce.h>
#include <loops/broadcasting.h>

namespace functions {
	namespace broadcast {
		template <typename X, typename Y, typename Z>
		class Broadcast;
	}

	namespace transform {
		template <typename X, typename Z>
		class Transform;
	}

	namespace scalar {
	}

	namespace reduce {
		template <typename X, typename Z>
		class ReduceFloatFunction;

        template <typename X>
        class ReduceSameFunction;
	}
}

namespace simdOps {

	template<typename T, typename Z>
	class Pooling2D {
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
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

			__shared__ int kH;
			__shared__ int kW;
			__shared__ int sH;
			__shared__ int sW;
			__shared__ int pH;
			__shared__ int pW;
			__shared__ int dH;
			__shared__ int dW;
			__shared__ int poolingMode;
			__shared__ T extraParam0;

			__shared__ int batchSize;
			__shared__ int inChannels;
			__shared__ int outH;
			__shared__ int outW;
			__shared__ int inH;
			__shared__ int inW;

            //__shared__ int *strideIn;
            //__shared__ int *strideOut;
            __shared__ int strideB;
            __shared__ int strideC;
            __shared__ int strideY;
            __shared__ int strideX;

			__shared__ int strideOB;
            __shared__ int strideOC;
            __shared__ int strideOY;
            __shared__ int strideOX;

            __shared__ int length;
            __shared__ int kHEff;
            __shared__ int kWEff;
			__shared__ bool fOrder;
		

			if (threadIdx.x == 0) {
				kH = (int)extraParams[0];
				kW = (int)extraParams[1];
				sH = (int)extraParams[2];
				sW = (int)extraParams[3];
				pH = (int)extraParams[4];
				pW = (int)extraParams[5];
				dH = (int)extraParams[6];			//Dilation, height dimension
				dW = (int)extraParams[7];			//Dilation, width dimension
				poolingMode = (int)extraParams[9];
				extraParam0 = extraParams[10];

				batchSize = shape::sizeAt(xShapeBuffer, 0);
				inChannels = shape::sizeAt(xShapeBuffer, 1);
				outH = shape::sizeAt(resultShapeBuffer, 2);
				outW = shape::sizeAt(resultShapeBuffer, 3);
				inH = shape::sizeAt(xShapeBuffer, 2);
				inW = shape::sizeAt(xShapeBuffer, 3);

            	strideB = shape::stride(xShapeBuffer)[0];
            	strideC = shape::stride(xShapeBuffer)[1];
            	strideY = shape::stride(xShapeBuffer)[2];
            	strideX = shape::stride(xShapeBuffer)[3];

				strideOB = shape::stride(resultShapeBuffer)[0];
            	strideOC = shape::stride(resultShapeBuffer)[1];
            	strideOY = shape::stride(resultShapeBuffer)[2];
            	strideOX = shape::stride(resultShapeBuffer)[3];

            	length = shape::length(resultShapeBuffer);

				//Replace kernel H/W with *effective* kernel H/W accounting for dilatyon
				kHEff = kH + (kH-1)*(dH-1);
				kWEff = kW + (kW-1)*(dW-1);

				fOrder = shape::order(resultShapeBuffer) == 'f';
/*
				if (blockIdx.x == 0) {
					printf("kH: %i; kW: %i; sH: %i; sW: %i; pH: %i; pW: %i; dH: %i; dW: %i; poolingMode: %i; extraParam0: %f;\n", kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, (float) extraParam0);
					printf("batchSize: %i; inChannels: %i; outH: %i; outW: %i; inH: %i; inW: %i; strideB: %i; strideC: %i; strideY: %i; strideX: %i;\n", batchSize, inChannels, outH, outW, inH, inW, strideB, strideC, strideY, strideX);
				}
*/
            }
            __syncthreads();

			int tid = blockIdx.x * gridDim.x + threadIdx.x;

            for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
				const int pw = index % outW;
    			const int ph = (index / outW) % outH;
    			const int c = (index / outW / outH) % inChannels;
    			const int n = index / outW / outH / inChannels;
    			int hstart = sH * ph - pH;
    			int wstart = sW * pw - pW;
    			int hend = hstart + kHEff;
    			int wend = wstart + kWEff;

//    			const int hSO = hstart;
//    			const int hEO = hend;

    			if(hstart < 0){
                    int f = (int)nd4j::math::nd4j_ceil<T>((T) -hstart / (T)dH);
                    hstart += f * dH;
                }
                if(wstart < 0){
                    int f = (int)nd4j::math::nd4j_ceil<T>((T) -wstart / (T) dW);
                    wstart += f * dW;
                }
                if(hend > inH){
                    int f = (int)nd4j::math::nd4j_ceil<T>((T) (hend-inH) / (T) dH);
                    hend -= f * dH;
                }
                if(wend > inW){
                    int f = (int)nd4j::math::nd4j_ceil<T>((T) (wend-inW) / (T) dW);
                    wend -= f * dW;
                }
    			int pool_size = (int)(nd4j::math::nd4j_ceil<T>((T) (hend-hstart) / (T) dH) * (int) nd4j::math::nd4j_ceil<T>((T) (wend-wstart) / (T) dW));	//Accounts for dilation

    			T sum = poolingMode == 0 ? -nd4j::DataTypeUtils::max<T>() : static_cast<T>(0.f);

    			T *input_slice = dx + (n * strideB + c * strideC);
    			if (poolingMode == 0) {
    			    for (int h = hstart; h < hend; h += dH) {
      				    for (int w = wstart; w < wend; w += dW) {
        				    T v = input_slice[h * strideY + w * strideX];
        				    if (v > sum)
        				        sum = v;
      				    }
    			    }
    			} else if (poolingMode == 1) {
    			    for (int h = hstart; h < hend; h += dH) {
      				    for (int w = wstart; w < wend; w += dW) {
        				    sum += input_slice[h * strideY + w * strideX];
      				    }
    			    }
    			} else if (poolingMode == 2) {
    			    for (int h = hstart; h < hend; h += dH) {
      				    for (int w = wstart; w < wend; w += dW) {
        				    sum += nd4j::math::nd4j_pow<T,T,T>(nd4j::math::nd4j_abs<T>(input_slice[h * strideY + w * strideX]), extraParam0);
      				    }
    			    }
    			}

				T res;

    			if (poolingMode == 0) {
                    res = sum;
    			} else if (poolingMode == 1) {
    			    int divide_factor = pool_size;  //Case 0: exclude padding
    			    if ((int) extraParam0 == 1)     //Case 1: include padding
					    divide_factor = kH * kW;

    			    res = sum / divide_factor;
    			} else if (poolingMode == 2) {
                    res = nd4j::math::nd4j_pow<T,T,T>(sum, (T) 1.0f / extraParam0);
    			}


				if (!fOrder) {
					result[index] = res;
                } else {
					result[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = res;
                }
/*
                if (index >= 0 && index < 400000) {
    			    printf("index: %i; hstart: %i; hend: %i; wstart: %i; wend: %i; ph: %i; pw: %i; hstart_orig: %i; hend_orig: %i;\n", index, hstart, hend, wstart, wend, ph, pw, hSO, hEO);
    			}
*/
            }
		}
#endif


static void execSpecial(T *in, Nd4jLong *inShapeBuffer, Z *out, Nd4jLong *outShapeBuffer, Z *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
	// input is  [bS, iC, iH, iW]
	// output is [bS, iC, oH, oW]

	const Nd4jLong kH = (int)extraParams[0];
	const Nd4jLong kW = (int)extraParams[1];
    const Nd4jLong sH = (int)extraParams[2];
    const Nd4jLong sW = (int)extraParams[3];
    const Nd4jLong pH = (int)extraParams[4];
    const Nd4jLong pW = (int)extraParams[5];    
    const Nd4jLong dH = (int)extraParams[6];
    const Nd4jLong dW = (int)extraParams[7];
    Nd4jLong poolingMode = (int)extraParams[9];
    T extraParam0 = extraParams[10];

    if(dH == 0 || dW == 0) {
       printf("Special_ops pooling2d:: dilation must not be zero, but got instead {%lld, %lld} \n", dH, dW);
       throw "";
    }

    const Nd4jLong kHEff = kH + (kH-1)*(dH-1);
    const Nd4jLong kWEff = kW + (kW-1)*(dW-1);

	const int bS = shape::sizeAt(inShapeBuffer, 0);
    const int iC = shape::sizeAt(inShapeBuffer, 1);
    const int iH = shape::sizeAt(inShapeBuffer, 2);
    const int iW = shape::sizeAt(inShapeBuffer, 3);
    const int oH = shape::sizeAt(outShapeBuffer, 2);
    const int oW = shape::sizeAt(outShapeBuffer, 3);            
    const Nd4jLong iStride0 = shape::stride(inShapeBuffer)[0];
    const Nd4jLong iStride1 = shape::stride(inShapeBuffer)[1];
    const Nd4jLong iStride2 = shape::stride(inShapeBuffer)[2];
    const Nd4jLong iStride3 = shape::stride(inShapeBuffer)[3];
    const Nd4jLong oStride0 = shape::stride(outShapeBuffer)[0];
    const Nd4jLong oStride1 = shape::stride(outShapeBuffer)[1];
    const Nd4jLong oStride2 = shape::stride(outShapeBuffer)[2];
    const Nd4jLong oStride3 = shape::stride(outShapeBuffer)[3];

    const Nd4jLong iStep2 = dH*iStride2;
    const Nd4jLong iStep3 = dW*iStride3;        
    const int kProd  = kH*kW;
    const T iStep2Inv = 1./iStep2; 
    const T iStep3Inv = 1./iStep3;

    Nd4jLong hstart, wstart, hend, wend;
    T sum, *pIn;

    if(poolingMode == 0) {        // max 
#pragma omp parallel for schedule(guided) private(pIn, sum, hstart, wstart, hend, wend)
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;
                        
                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;                        
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;
                        
                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));                            
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));                            

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = -MAX_FLOAT;
                                                                    
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3) {
                                T val = pIn[kh + kw];
                                    if (val > sum)
                                        sum = val;
                                    }
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }    
    }
/*************************************************************************/    
    else if(poolingMode == 1) {      // avg
#pragma omp parallel for schedule(guided) private(pIn, sum, hstart, wstart, hend, wend)        
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = static_cast<T>(0.);
                                            
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += pIn[kh + kw];
                                
                        if ((int) extraParam0 == 0)         //Exclude padding
                            sum /= static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(hend-hstart) / static_cast<double>(iStep2))) * static_cast<T>(nd4j::math::nd4j_ceil<double,T>(static_cast<double>(wend-wstart) / static_cast<double>(iStep3)));   //Accounts for dilation
                        else if ((int) extraParam0 == 1)    //Include padding
                            sum /= kProd;
                    
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }
    }    
/*************************************************************************/    
    else if(poolingMode == 2) {  // pnorm
#pragma omp parallel for schedule(guided) private(pIn, sum, hstart, wstart, hend, wend)    
        for(int b = 0; b < bS; ++b) {
            for(int c = 0; c < iC; ++c) {                                                            
                for(int oh = 0; oh < oH; ++oh) {
                    for(int ow = 0; ow < oW; ++ow) {
                        
                        pIn  = in  + b * iStride0 + c * iStride1;

                        hstart = oh * sH - pH;
                        wstart = ow * sW - pW;
                        hend = hstart + kHEff;
                        wend = wstart + kWEff;

                        if(hstart < 0)
                            hstart += dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-hstart) / static_cast<T>(dH));
                        if(wstart < 0)
                            wstart += dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(-wstart) / static_cast<T>(dW));
                        if(hend > iH)
                            hend -= dH * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(hend-iH) / static_cast<T>(dH));
                        if(wend > iW)
                            wend -= dW * (Nd4jLong)nd4j::math::nd4j_ceil<T>(static_cast<T>(wend-iW) / static_cast<T>(dW));

                        hstart *= iStride2;
                        hend   *= iStride2;
                        wstart *= iStride3;
                        wend   *= iStride3;

                        sum = static_cast<T>(0.);
                                                                    
                        for (Nd4jLong kh = hstart; kh < hend; kh += iStep2) 
                            for (Nd4jLong kw = wstart; kw < wend; kw += iStep3)
                                sum += nd4j::math::nd4j_pow<T, T, T>(nd4j::math::nd4j_abs<T>(pIn[kh + kw]), extraParam0);
                                
                        sum = nd4j::math::nd4j_pow<T,T,T>(sum, (T) 1. / extraParam0);
                                                          
                        out[b * oStride0 + c * oStride1 + oh * oStride2 + ow * oStride3] = sum;
                    }
                }
            }
        }
    }
    else {
        nd4j_printf("Special_ops::pooling2d: pooling mode argument can take three values only: 0, 1, 2, but got %i instead !\n", poolingMode);
        throw "";
	}
}

		op_def static T op(T d1, Z *params) {
			return d1;
		}


		/** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
		*  normally negative indices are bad, OK here because of other checks on input indices
		*  Uses unrolled loop specifically for length 4
		*/
		static _CUDA_HD int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
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
		static _CUDA_HD int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};


    FORCEINLINE bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

	template<typename T>
	class 
	Im2col {
	public:
		static const bool requiresSpecial = true;

		static _CUDA_HD int outSize(int size, int k, int s, int p, bool coverAll) {
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
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/
			int kernelHeight = (int)extraParams[0];
			int kernelWidth = (int)extraParams[1];
			int strideY = (int)extraParams[2];
			int strideX = (int)extraParams[3];
			int padHeight = (int)extraParams[4];
			int padWidth = (int)extraParams[5];
			int dY = (int)extraParams[6];			//Dilation, height/y dimension
			int dX = (int)extraParams[7];			//Dilation, width/x dimension
			int kSize = kernelWidth * kernelHeight;
			T zeroPadVal = (T)extraParams[9];	//Value to use when value is padding. Usually 0 but not always

			auto outShape = shape::shapeOf(resultShapeBuffer);
			auto resultOrder = shape::order(resultShapeBuffer);
			auto outStride = shape::stride(resultShapeBuffer);

			auto inShape = shape::shapeOf(xShapeBuffer);
			auto inStride = shape::stride(xShapeBuffer);

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
						int h_im = h_offset + i * dY;
						int w_im = w_offset + j * dX;
						int i_f = 0;
						int i_c_temp = i_c;
						for (int dim = 5; dim >= 0; dim--) {
							i_f += (i_c_temp % outShape[dim])  * outStride[dim];
							i_c_temp = i_c_temp / outShape[dim];
						}
						if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width){
							result[i_f] = data_im_ptr[i * dY * strideh + j * dX * stridew];
						} else result[i_f] = zeroPadVal;

						//result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
						data_col_ptr += height_col * width_col;
						i_c += height_col * width_col;
					}
				}
			}
		}
#endif


		static void execSpecial(
			T *imBuff,
			Nd4jLong *imShapeBuffer,
			T *colBuff,
			Nd4jLong *colShapeBuffer,
			T *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			/*kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1], 0, false*/

			// [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]        

			int kH = (int)extraParams[0];
			int kW = (int)extraParams[1];
			int sH = (int)extraParams[2];
			int sW = (int)extraParams[3];
			int pH = (int)extraParams[4];
			int pW = (int)extraParams[5];
			int dH = (int)extraParams[6];			//Dilation, height/y dimension
			int dW = (int)extraParams[7];			//Dilation, width/x dimension            
            T zeroPadVal = extraParams[9];

            auto colShape  = shape::shapeOf(colShapeBuffer);
            auto colStride = shape::stride(colShapeBuffer);
            auto imShape = shape::shapeOf(imShapeBuffer);
            auto imStride = shape::stride(imShapeBuffer);

            const int bS = imShape[0];
            const int iC = imShape[1];
            const int iH = imShape[2];
            const int iW = imShape[3];
            const int oH = colShape[4];
            const int oW = colShape[5];
            const Nd4jLong colStride0 = colStride[0];
            const Nd4jLong colStride1 = colStride[1];
            const Nd4jLong colStride2 = colStride[2];
            const Nd4jLong colStride3 = colStride[3];
            const Nd4jLong colStride4 = colStride[4];
            const Nd4jLong colStride5 = colStride[5];
            const Nd4jLong imStride0  = imStride[0];
            const Nd4jLong imStride1  = imStride[1];
            const Nd4jLong imStride2  = imStride[2];
            const Nd4jLong imStride3  = imStride[3];

            T *col, *im;
            int imRow, imCol;
            
            if (shape::order(imShapeBuffer) == 'c' &&  shape::order(colShapeBuffer) == 'c' && shape::strideDescendingCAscendingF(imShapeBuffer) && shape::strideDescendingCAscendingF(colShapeBuffer)) {

#pragma omp parallel for schedule(static) proc_bind(close) private(col, im, imRow, imCol)
                for (int b = 0; b < bS; b++) {
                    for (int c = 0; c < iC; ++c) {        
                        for (int kRow = 0; kRow < kH; ++kRow) {                        
                            for (int kCol = 0; kCol < kW; ++kCol) {                            
                                for (int colH = 0; colH < oH; ++colH) {
                                    for (int colW = 0; colW < oW; ++colW) {                    
                                
                                        imRow = (-pH + kRow * dH) + colH*sH;
                                        imCol = (-pW + kCol * dW) + colW*sW;
                                        
                                        col = colBuff + b*colStride0 + c*colStride1 + kRow*colStride2 + kCol*colStride3 + colH*colStride4 + colW*colStride5;
                                        im  = imBuff  + b*imStride0  + c*imStride1  + imRow*imStride2 + imCol*imStride3; 
                                                    
                                        if (static_cast<unsigned>(imRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(imCol) >= static_cast<unsigned>(iW))
                                            *col = zeroPadVal;
                                        else 
                                            *col = *im;
                                    }
                                }
                            }
                        }
                    }
                }  
            }
            else {
 
#pragma omp parallel for schedule(static) proc_bind(close) private(im, col, imRow, imCol)    
                for (int b = 0; b < bS; b++) {
                    for (int colH = 0; colH < oH; ++colH) {
                        for (int colW = 0; colW < oW; ++colW) {
                            for (int c = 0; c < iC; ++c) {
                                for (int kRow = 0; kRow < kH; ++kRow) {                        
                                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                        imRow = (-pH + kRow * dH) + colH*sH;
                                        imCol = (-pW + kCol * dW) + colW*sW;
                                        
                                        col = colBuff + b*colStride0 + c*colStride1 + kRow*colStride2 + kCol*colStride3 + colH*colStride4 + colW*colStride5;
                                        im  = imBuff  + b*imStride0  + c*imStride1  + imRow*imStride2 + imCol*imStride3;
                                                    
                                        if (static_cast<unsigned>(imRow) >= static_cast<unsigned>(iH) || static_cast<unsigned>(imCol) >= static_cast<unsigned>(iW))
                                            *col = zeroPadVal;
                                        else 
                                            *col = *im;
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
		static _CUDA_HD int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
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
		static _CUDA_HD int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};

	template<typename T, typename Z>
	class Histogram {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		static inline __device__ void execSpecialCuda(
			T *dx,
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

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
				Nd4jLong *xShapeBuffer,
				Z *result,
				Nd4jLong *resultShapeBuffer,
				Z *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

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


        op_def static T op(T d1, Z *params) {
            return d1;
        }
	};

	template<typename X>
	class Col2Im {

	public:
		static const bool requiresSpecial = true;
#ifdef __CUDACC__
		/**
		* https://github.com/pjreddie/darknet/blob/master/src/col2im_kernels.cu
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			auto inShape = shape::shapeOf(xShapeBuffer);
			auto inStride = shape::stride(xShapeBuffer);

			int strideex = inStride[0];
			int stridech = inStride[1];
			int stridekrow = inStride[2];
			int stridekcol = inStride[3];
			int striderow = inStride[4];
			int stridecol = inStride[5];

			int kernelHeight = inShape[2];
			int kernelWidth = inShape[3];

			// C

			int strideY = (int)extraParams[0];
			int strideX = (int)extraParams[1];
            int padHeight = (int)extraParams[2];
			int padWidth = (int)extraParams[3];
            int imgHeight = (int)extraParams[4];
            int imgWidth = (int)extraParams[5];
			int dY = (int)extraParams[6];			//Dilation in height/y dimension
            int dX = (int)extraParams[7];			//Dilation in width/x dimension

			auto outShape = shape::shapeOf(resultShapeBuffer);
			auto resultOrder = shape::order(resultShapeBuffer);
			auto outStride = shape::stride(resultShapeBuffer);

			int samples = outShape[0];
			int depth = outShape[1];
			int imgH = outShape[2];
			int imgW = outShape[3];

			int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
			int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

			int n = samples * depth * imgHeight * imgWidth;

			/*if (threadIdx.x == 0)
			printf("Kernel h: [%i], w: [%i]; Col h: [%i], w: [%i]; Stride x: [%i], y: [%i]; Height: [%i], Width: [%i], Depth: [%i], N: [%i], Samples: [%i]\n",
			kernelHeight, kernelWidth, height_col, width_col, strideX, strideY, imgHeight, imgWidth, depth, n, samples);*/

			//Effective kernel size, accounting for dilation
			int kEffectiveW = kernelWidth + (kernelWidth - 1) * (dX - 1);
			int kEffectiveH = kernelHeight + (kernelHeight - 1) * (dY - 1);

			for (int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
				T val = 0;
				int w_im = i % imgWidth + padWidth;
				int h_im = (i / imgWidth) % imgHeight + padHeight;
				int c_im = i / (imgWidth * imgHeight);

				int num_im = c_im / depth;
				int depth_im = c_im % depth;

				// compute the start and end of the output
				// These are the indexes for dimensions ??? in the 6d col matrix
				int w_col_start = (w_im < kEffectiveW) ? 0 : (w_im - kEffectiveW) / strideX + 1;
				int w_col_end = nd4j::math::nd4j_min<int>(w_im / strideX + 1, width_col);

				int h_col_start = (h_im < kEffectiveH) ? 0 : (h_im - kEffectiveH) / strideY + 1;
				int h_col_end = nd4j::math::nd4j_min<int>(h_im / strideY + 1, height_col);


				//Iterate over col entries in the 6d array... these are added up
				for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
					for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
						int h_k = (h_im - h_col * strideY);
						int w_k = (w_im - w_col * strideX);
						
						if(h_k % dY == 0 && w_k % dX == 0){
							h_k /= dY;
							w_k /= dX;

							int data_col_index = num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;
							val += dx[data_col_index];
						}
					}
				}
				int i_f = 0;
				int i_c = i;
				for (int dim = 3; dim >= 0; dim--)
				{
					i_f += (i_c % outShape[dim])  * outStride[dim];
					i_c = i_c / outShape[dim];
				}
				result[i_f] = val;
			}
		}
#endif

		static void execSpecial(
			X *colBuff,
			Nd4jLong *colShapeBuffer,
			X *imBuff,
			Nd4jLong *imShapeBuffer,
			X *extraParams,
			Nd4jLong *tadShapeInfo,
			Nd4jLong *tadOffsets) {

            // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

            auto colShape  = shape::shapeOf(colShapeBuffer);
            auto colStride = shape::stride(colShapeBuffer);
            auto imShape = shape::shapeOf(imShapeBuffer);
            auto imStride = shape::stride(imShapeBuffer);            

            const int sH = (int)extraParams[0];
            const int sW = (int)extraParams[1];
            const int pH = (int)extraParams[2];
            const int pW = (int)extraParams[3];
            const int iH = (int)extraParams[4];
            const int iW = (int)extraParams[5];
            const int dH = (int)extraParams[6];     
            const int dW = (int)extraParams[7];     

            const int bS = imShape[0];
            const int iC = imShape[1];
            const int kH = colShape[2];
            const int kW = colShape[3];                    
            const int oH = colShape[4];
            const int oW = colShape[5];
            const Nd4jLong colStride0 = colStride[0];
            const Nd4jLong colStride1 = colStride[1];
            const Nd4jLong colStride2 = colStride[2];
            const Nd4jLong colStride3 = colStride[3];
            const Nd4jLong colStride4 = colStride[4];
            const Nd4jLong colStride5 = colStride[5];
            const Nd4jLong imStride0  = imStride[0];
            const Nd4jLong imStride1  = imStride[1];
            const Nd4jLong imStride2  = imStride[2];
            const Nd4jLong imStride3  = imStride[3];

            // initial zeroing of image content
            const Nd4jLong imEWS = nd4j::math::nd4j_abs<Nd4jLong>(shape::elementWiseStride(imShapeBuffer));
            if(imEWS == 1)
                 memset(imBuff, 0, shape::length(imShapeBuffer) * sizeof(X));
            else 
#pragma omp parallel for schedule(static) proc_bind(close)
                for (int i = 0; i < shape::length(imShapeBuffer) * imEWS; i += imEWS) 
                    imBuff[i] = static_cast<X>(0.f);
            

            X *col, *im;
            int imRow, imCol;

            if (shape::order(colShapeBuffer) == 'c' &&  shape::order(imShapeBuffer) == 'c' && shape::strideDescendingCAscendingF(colShapeBuffer) && shape::strideDescendingCAscendingF(imShapeBuffer)) {
            
#pragma omp parallel for schedule(static) proc_bind(close) private(col, im, imRow, imCol)
                for (int b = 0; b < bS; b++) {        
                    for (int c = 0; c < iC; ++c) {                    
                        for (int kRow = 0; kRow < kH; ++kRow) {                        
                            for (int kCol = 0; kCol < kW; ++kCol) {                            
                                for (int colH = 0; colH < oH; ++colH) {
                                    for (int colW = 0; colW < oW; ++colW) {                    

                                        imRow = (-pH + kRow * dH) + colH*sH;
                                        imCol = (-pW + kCol * dW) + colW*sW;

                                        col = colBuff + b*colStride0 + c*colStride1 + kRow*colStride2 + kCol*colStride3 + colH*colStride4 + colW*colStride5;
                                        im  = imBuff  + b*imStride0  + c*imStride1  + imRow*imStride2 + imCol*imStride3;

                                        if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(imCol) < static_cast<unsigned>(iW))
                                            *im += *col;
                                    }
                                }
                            }
                        }
                    }
                }  
            }
            else {

#pragma omp parallel for schedule(static) proc_bind(close) private(im, col, imRow, imCol)
                for (int b = 0; b < bS; b++) {        
                    for (int colH = 0; colH < oH; ++colH) {
                        for (int colW = 0; colW < oW; ++colW) {
                            for (int c = 0; c < iC; ++c) {                        
                                for (int kRow = 0; kRow < kH; ++kRow) {                        
                                    for (int kCol = 0; kCol < kW; ++kCol) {                            
                        
                                        imRow = (-pH + kRow * dH) + colH*sH;
                                        imCol = (-pW + kCol * dW) + colW*sW;
                                        
                                        col = colBuff + b*colStride0 + c*colStride1 + kRow*colStride2 + kCol*colStride3 + colH*colStride4 + colW*colStride5;
                                        im  = imBuff  + b*imStride0  + c*imStride1  + imRow*imStride2 + imCol*imStride3;

                                        if (static_cast<unsigned>(imRow) < static_cast<unsigned>(iH) && static_cast<unsigned>(imCol) < static_cast<unsigned>(iW))
                                            *im += *col;
                                    }
                                }
                            }
                        }                           
                    }
                }  
            }
        }

		op_def static X op(X d1, X *params) {
			return d1;
		}


		/** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
		*  normally negative indices are bad, OK here because of other checks on input indices
		*  Uses unrolled loop specifically for length 4
		*/
		static _CUDA_HD int getOffsetUnsafe4(int baseOffset, int *shape, int *stride, int *indices) {
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
		static _CUDA_HD int getOffsetUnsafe6(int baseOffset, int *shape, int *stride, int *indices) {
			int offset = baseOffset;
			if (shape[0] != 1) offset += indices[0] * stride[0];
			if (shape[1] != 1) offset += indices[1] * stride[1];
			if (shape[4] != 1) offset += indices[4] * stride[4];
			if (shape[5] != 1) offset += indices[5] * stride[5];
			return offset;
		}

	};


	template<typename X>
	class Reverse {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		static inline __device__ void execSpecialCuda(T *dx, Nd4jLong *xShapeBuffer, T *result, Nd4jLong *zShapeBuffer, T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
            __shared__ Nd4jLong xLength;
			__shared__ int xEWS;
            __shared__ char xOrder;
            __shared__ Nd4jLong sLength;
            __shared__ T *shmem;
            int tid = threadIdx.x + blockIdx.x * blockDim.x;

            if (threadIdx.x == 0) {
                xLength = shape::length(xShapeBuffer);
			    xEWS = shape::elementWiseStride(xShapeBuffer);
                xOrder = shape::order(xShapeBuffer);
                sLength = xLength - 1;

                extern __shared__ unsigned char shrd[];
                shmem = (T *) shrd;
            }
            __syncthreads();



            if (dx == result) {

                if (xEWS == 1) {
                    for (int e = tid; e < xLength / 2; e += blockDim.x * gridDim.x) {
                        Nd4jLong idx = sLength - e;
                        T tmp = dx[e];
                        dx[e] = dx[idx];
                        dx[idx] = tmp;
                    }
                } else if (xEWS >= 1) {
                    for (int e = tid; e < xLength / 2; e += blockDim.x * gridDim.x) {
                        Nd4jLong idx1 = (sLength - e) * xEWS;
                        Nd4jLong idx2 =  e * xEWS;
                        T tmp = dx[idx2];
                        dx[idx2] = dx[idx1];
                        dx[idx1] = tmp;
                    }
                } else {
                    __shared__ int xRank;
                    __shared__ Nd4jLong *xShape;
                    __shared__ Nd4jLong *xStride;

                    if (threadIdx.x == 0) {
				        xRank = shape::rank(xShapeBuffer);
                        xShape = shape::shapeOf(xShapeBuffer);
                        xStride = shape::stride(xShapeBuffer);
				    }
				    __syncthreads();

					Nd4jLong xCoord[MAX_RANK];
					Nd4jLong zCoord[MAX_RANK];

					for (int e = tid; e < xLength / 2; e += blockDim.x * gridDim.x) {
                        if (xOrder == 'c') {
                            shape::ind2subC(xRank, xShape, e, xCoord);
                            shape::ind2subC(xRank, xShape, sLength - e, zCoord);
                        } else {
                            shape::ind2sub(xRank, xShape, e, xCoord);
                            shape::ind2sub(xRank, xShape, sLength - e, zCoord);
                        }

                        auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto zOffset = shape::getOffset(0, xShape, xStride, zCoord, xRank);

                        result[zOffset] = dx[xOffset];
					}
                }

            } else {
                __shared__ int zEWS;
				__shared__ char zOrder;

				if (threadIdx.x == 0) {
				    zEWS = shape::elementWiseStride(zShapeBuffer);
				    zOrder = shape::order(zShapeBuffer);
				}
				__syncthreads();

                if (xEWS == 1 && zEWS == 1 && xOrder == zOrder) {
                    // loop for whole array
                    for (int e = tid; e < xLength; e += blockDim.x * gridDim.x) {
                        result[sLength - e] = dx[e];
                    }
                } else if (xEWS >= 1 && zEWS >= 1 && xOrder == zOrder) {

                    for (int e = tid; e < xLength; e += blockDim.x * gridDim.x) {
                        result[(sLength - e) * zEWS] = dx[e * xEWS];
                    }
                } else {
                    __shared__ int xRank;
                    __shared__ Nd4jLong *xShape;
                    __shared__ Nd4jLong *xStride;

					__shared__ int zRank;
					__shared__ Nd4jLong *zShape;
                    __shared__ Nd4jLong *zStride;

                    if (threadIdx.x == 0) {
				        xRank = shape::rank(xShapeBuffer);
                        xShape = shape::shapeOf(xShapeBuffer);
                        xStride = shape::stride(xShapeBuffer);

					    zRank = shape::rank(zShapeBuffer);
					    zShape = shape::shapeOf(zShapeBuffer);
                        zStride = shape::stride(zShapeBuffer);
				    }
				    __syncthreads();

					Nd4jLong xCoord[MAX_RANK];
					Nd4jLong zCoord[MAX_RANK];

                    for (int e = tid; e < xLength; e += blockDim.x * gridDim.x) {
                        if (xOrder == 'c') {
                            shape::ind2subC(xRank, xShape, e, xCoord);
                            shape::ind2subC(xRank, xShape, sLength - e, zCoord);
                        } else {
                            shape::ind2sub(xRank, xShape, e, xCoord);
                            shape::ind2sub(xRank, xShape, sLength - e, zCoord);
                        }


                        auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto zOffset = shape::getOffset(0, xShape, xStride, zCoord, xRank);

                        result[zOffset] = dx[xOffset];
                    }
                }
            }
		}

#endif


		static void execSpecial(X *dx, Nd4jLong *xShapeBuffer, X *result, Nd4jLong *zShapeBuffer, X *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			Nd4jLong xLength = shape::length(xShapeBuffer);
			int xEWS = shape::elementWiseStride(xShapeBuffer);
            char xOrder = shape::order(xShapeBuffer);
            Nd4jLong sLength = xLength - 1;

			// two step phase here
			if (dx == result) {
				if (xEWS == 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < xLength / 2; e++) {
                        Nd4jLong idx = sLength - e;
                        auto tmp = dx[e];
                        dx[e] = dx[idx];
                        dx[idx] = tmp;
                    }
				} else if (xEWS > 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < xLength / 2; e++) {
                        Nd4jLong idx1 = (sLength - e) * xEWS;
                        Nd4jLong idx2 =  e * xEWS;
                        auto tmp = dx[idx2];
                        dx[idx2] = dx[idx1];
                        dx[idx1] = tmp;
                    }
				} else {
                    int xRank = shape::rank(xShapeBuffer);
                    auto xShape = shape::shapeOf(xShapeBuffer);
                    auto xStride = shape::stride(xShapeBuffer);

                    Nd4jLong xCoord[MAX_RANK];
                    Nd4jLong zCoord[MAX_RANK];

#pragma omp parallel for private(xCoord, zCoord) schedule(guided)
                    for (Nd4jLong e = 0; e < xLength / 2; e++) {
                        if (xOrder == 'c') {
                            shape::ind2subC(xRank, xShape, e, xCoord);
                            shape::ind2subC(xRank, xShape, sLength - e, zCoord);
                        } else {
                            shape::ind2sub(xRank, xShape, e, xCoord);
                            shape::ind2sub(xRank, xShape, sLength - e, zCoord);
                        }

                        auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto zOffset = shape::getOffset(0, xShape, xStride, zCoord, xRank);

                        result[zOffset] = dx[xOffset];
                    }
				}
			} else {
				// single step phase here
				auto zEWS = shape::elementWiseStride(zShapeBuffer);
				auto zOrder = shape::order(zShapeBuffer);

				if (xEWS == 1 && zEWS == 1 && xOrder == zOrder) {
#pragma omp parallel for schedule(guided)
					for (Nd4jLong e = 0; e < xLength; e++) {
						result[sLength - e] = dx[e];
					}
				} else if (xEWS >= 1 && zEWS >= 1 && xOrder == zOrder) {
#pragma omp parallel for schedule(guided)
					for (Nd4jLong e = 0; e < xLength; e++) {
						result[(sLength - e) * zEWS] = dx[e * xEWS];
					}
				} else {

					auto xRank = shape::rank(xShapeBuffer);
                    auto xShape = shape::shapeOf(xShapeBuffer);
                    auto xStride = shape::stride(xShapeBuffer);

					auto zRank = shape::rank(zShapeBuffer);
					auto zShape = shape::shapeOf(zShapeBuffer);
                    auto zStride = shape::stride(zShapeBuffer);

					Nd4jLong xCoord[MAX_RANK];
					Nd4jLong zCoord[MAX_RANK];

#pragma omp parallel for private(xCoord, zCoord) schedule(guided)
					for (Nd4jLong e = 0; e < xLength; e++) {

						if (xOrder == 'c')
							shape::ind2subC(xRank, xShape, e, xCoord);
						else
							shape::ind2sub(xRank, xShape, e, xCoord);

						if (zOrder == 'c')
                            shape::ind2subC(zRank, zShape, (sLength - e), zCoord);
                        else
                        	shape::ind2sub(zRank, zShape, (sLength - e), zCoord);

						auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

						result[zOffset] = dx[xOffset];
					}
				}
			}
		}

        op_def static X op(X d1, X *params) {
            return d1;
        }
	};

	template<typename X, typename Z>
	class SoftMax {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

			auto shape = shape::shapeOf(xShapeBuffer);
			__shared__ T maxResult;
			__shared__ Nd4jLong *maxResultShapeBuffer;

			auto length = shape::length(xShapeBuffer);

			auto stride = shape::stride(xShapeBuffer);
			//compute the row wise maxes

			__shared__ Nd4jLong maxShape[2];

			// it's always 2d here
			__shared__ Nd4jLong tempBuffer[8];

			if (threadIdx.x == 0) {
			    maxResult = (T) 0.0;
			    maxShape[0] = shape[0];
			    maxShape[1] = 1;
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);
			}
			__syncthreads();

			functions::reduce::ReduceFloatFunction<T>::template execScalarCuda<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Subtract<T>>(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			functions::transform::Transform<T>::template transformCuda<simdOps::Exp<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);
			__syncthreads();

			//take the sum for the exponential
			functions::reduce::ReduceFloatFunction<T>::template execScalarCuda<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Divide<T>>(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);

		}
#endif

		static void execSpecial(
			X *vx,
			Nd4jLong *xShapeBuffer,
			Z *vresult,
			Nd4jLong *resultShapeBuffer,
			Z *vextraParams,
			Nd4jLong *tadShapeInfo,
			Nd4jLong *tadOffsets) {
		    auto dx = reinterpret_cast<X *>(vx);
		    auto result = reinterpret_cast<X *>(vresult);
		    auto extraParams = reinterpret_cast<Z *>(vextraParams);

			if (shape::isMatrix(xShapeBuffer)) {
				auto shape = shape::shapeOf(xShapeBuffer);
				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				//compute the row wise maxes
				std::vector<X> maxResult(shape[0]);
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;
				Nd4jLong maxShape[2] = { shape[0], 1 };
				auto maxResultShapeBuffer = shape::shapeBuffer(2, nd4j::ArrayOptions::dataType(xShapeBuffer), maxShape);
				functions::reduce::ReduceSameFunction<X>::template exec<simdOps::Max<X>>(reinterpret_cast<void *>(dx), xShapeBuffer, reinterpret_cast<void *>(extraParams), reinterpret_cast<void *>(maxResult.data()), maxResultShapeBuffer, maxDimension, 1,  nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<X, X, Z>::template exec<simdOps::Subtract<X,X,Z>>(dx, xShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, nullptr, nullptr, nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<X,Z>::template exec<simdOps::Exp<X,Z>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);

				//take the sum for the exponential
				functions::reduce::ReduceSameFunction<Z>::template exec<simdOps::Sum<Z>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1, nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<Z,X,Z>::template exec<simdOps::Divide<Z,X,Z>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, nullptr, nullptr, nullptr, nullptr);

				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer)) {
				auto max = -nd4j::DataTypeUtils::max<X>();
				X sum = 0;
				int elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);
				int length = shape::length(xShapeBuffer);
				if (elementWiseStride >= 1 && resultElementWiseStride >= 1) {
					if (elementWiseStride == 1 && resultElementWiseStride == 1) {

#pragma omp simd reduction(maxT:max)
						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<X>(max, dx[i]);
						}

#pragma omp parallel for simd reduction(sumT:sum)
						for (int i = 0; i < length; i++) {
                            result[i] = nd4j::math::nd4j_exp<X,X>(dx[i] - max);
							sum += result[i];
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i] /= sum;
						}
					}
					else {

#pragma omp simd reduction(maxT:max)
						for (int i = 0; i < length; i++) {
							max = nd4j::math::nd4j_max<X>(max, dx[i * elementWiseStride]);
						}

#pragma omp parallel for simd reduction(sumT:sum)
						for (int i = 0; i < length; i++) {
                            auto r = nd4j::math::nd4j_exp<X, X>(dx[i * elementWiseStride] - max);
                            result[i * resultElementWiseStride] = r;
							sum += r;
						}

#pragma omp simd
						for (int i = 0; i < length; i++) {
							result[i * resultElementWiseStride] /= sum;
						}
					}
				}
			}
		}

		op_def static X op(X d1, Z *params) {
			return nd4j::math::softplus<X>(d1);
		}
	};



	template<typename X, typename Z>
	class LogSoftMax {
	public:
		static const bool requiresSpecial = true;
#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			auto shape = shape::shapeOf(xShapeBuffer);
			auto stride = shape::stride(xShapeBuffer);
			//iterate along rows

			__shared__ T maxResult;
			__shared__ Nd4jLong *maxResultShapeBuffer;
			if (threadIdx.x == 0) {

				maxResult = (T) 0.0;
			}
			__syncthreads();
			//compute the row wise maxes

			Nd4jLong maxShape[2] = { shape[0], 1 };
			__shared__ Nd4jLong tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);
			__syncthreads();

			functions::reduce::ReduceFloatFunction<T>::template execScalarCuda<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Subtract<T>>(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			functions::transform::Transform<T>::template transformCuda<simdOps::Exp<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);
			__syncthreads();

			//take the sum for the exponential
			functions::reduce::ReduceFloatFunction<T>::template execScalarCuda<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//divide by the sum
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Divide<T>>(maxResult, result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			functions::transform::Transform<T>::template transformCuda<simdOps::Log<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);

		}
#endif


		static void execSpecial(
			X *dx,
			Nd4jLong *xShapeBuffer,
			Z *result,
			Nd4jLong *resultShapeBuffer,
			Z *extraParams,
			Nd4jLong *tadShapeInfo,
			Nd4jLong *tadOffsets) {

			if (shape::isMatrix(xShapeBuffer, 2)) {
				auto shape = shape::shapeOf(xShapeBuffer);
				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				//compute the row wise maxes
				std::vector <X> maxResult(shape[0]);

#pragma omp simd
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;

				Nd4jLong maxShape[2] = { shape[0], 1 };
				auto maxResultShapeBuffer = shape::shapeBuffer(2, nd4j::ArrayOptions::dataType(xShapeBuffer), maxShape);
				functions::reduce::ReduceSameFunction<X>::template exec<simdOps::Max<X>>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1, nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<X,X,Z>::template exec<simdOps::Subtract<X,X,Z>>(dx, xShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, nullptr, nullptr, nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<Z,Z>::template exec<simdOps::Exp<Z,Z>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);

				//take the sum for the exponential
				functions::reduce::ReduceSameFunction<Z>::template exec<simdOps::Sum<Z>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1, nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<Z,X,Z>::template exec<simdOps::Divide<Z,X,Z>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, nullptr, nullptr, nullptr, nullptr);

				functions::transform::Transform<Z,Z>::template exec<simdOps::Log<Z,Z>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);


				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				auto max = -FLOAT_MAX_VALUE;
				X sum = 0;

				auto elementWiseStride = shape::elementWiseStride(xShapeBuffer);
                auto length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {
#pragma omp simd reduction(maxT:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<X>(max, result[i]);
					}

#pragma omp simd reduction(sumT:sum)
					for (int i = 0; i < length; i++) {
						result[i] = nd4j::math::nd4j_exp<X, X>(dx[i] - max);
						sum += result[i];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
						result[i] = nd4j::math::nd4j_log<X, X>(result[i]);
					}
				}
				else if (elementWiseStride > 1) {
#pragma omp simd reduction(maxT:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<X>(max, result[i * elementWiseStride]);
					}

#pragma omp simd reduction(sumT:sum)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<X, X>(dx[i * elementWiseStride] - max);
						sum += result[i * elementWiseStride];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
						result[i * elementWiseStride] = nd4j::math::nd4j_log<X, X>(result[i * elementWiseStride]);
					}
				}
			}
		}

		op_def static X op(X d1, Z *params) {
			return nd4j::math::softplus<X>(d1);
		}
	};


	/**
	* softmax(x)
	*/
	template<typename X, typename Z>
	class SoftMaxDerivative {
	public:
		static const bool requiresSpecial = true;

#ifdef __CUDACC__
		/**
		*
		*/

		static inline __device__ void execSpecialCuda(
			T *dx,
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

			auto shape = shape::shapeOf(xShapeBuffer);
			__shared__ T maxResult;
			__shared__ Nd4jLong *maxResultShapeBuffer;
			__shared__ Nd4jLong resultEWS;

			auto length = shape::length(xShapeBuffer);

			if (threadIdx.x == 0) {
				resultEWS = shape::elementWiseStride(resultShapeBuffer);

				maxResult = (T) 0.0;
			}
			__syncthreads();

			auto tride = shape::stride(xShapeBuffer);
			Nd4jLong maxShape[2] = { shape[0], 1 };

			__shared__ Nd4jLong tempBuffer[8];

			if (threadIdx.x == 0)
				maxResultShapeBuffer = shape::shapeBuffer(2, maxShape, tempBuffer);
			__syncthreads();

			functions::reduce::ReduceFloatFunction<T>::template execScalarCuda<simdOps::Max<T>>(dx, xShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
			__syncthreads();

			//subtract max of each row
			functions::scalar::ScalarTransform<T>::template transformCuda<simdOps::Subtract<T>>(maxResult, dx, xShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, manager);
			__syncthreads();

			//after subtracting the row wise maxes take the exp
			functions::transform::Transform<T>::template transformCuda<simdOps::Exp<T>>(result, resultShapeBuffer, extraParams, result, resultShapeBuffer, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);
			__syncthreads();

			//take the sum for the exponential
			functions::reduce::ReduceFloatFunction<T>::template execScalarCuda<simdOps::Sum<T>>(result, resultShapeBuffer, extraParams, &maxResult, maxResultShapeBuffer, reductionPointer, manager, nullptr);
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
			X *dx,
			Nd4jLong *xShapeBuffer,
			Z *result,
			Nd4jLong *resultShapeBuffer,
			Z *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			if (shape::isMatrix(xShapeBuffer, 2)) {
				auto shape = shape::shapeOf(xShapeBuffer);

				auto resultEleStide = shape::elementWiseStride(resultShapeBuffer);

				//iterate along rows
				int dimension[1] = { 0 };
				int maxDimension[1] = { 1 };
				auto len = shape::length(xShapeBuffer);
				//compute the row wise maxes
				std::vector <X> maxResult(shape[0]);
#pragma omp simd
				for (int i = 0; i < shape[0]; i++)
					maxResult[i] = 0.0;

				Nd4jLong maxShape[2] = { shape[0], 1 };
				auto maxResultShapeBuffer = shape::shapeBuffer(2, nd4j::ArrayOptions::dataType(xShapeBuffer), maxShape);
				functions::reduce::ReduceSameFunction<X>::template exec<simdOps::Max<X>>(dx, xShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1, nullptr, nullptr);

				//subtract max of each row
				functions::broadcast::Broadcast<Z,X,Z>::template exec<simdOps::Subtract<Z,X,Z>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, nullptr, nullptr, nullptr, nullptr);

				//after subtracting the row wise maxes take the exp
				functions::transform::Transform<Z,Z>::template exec<simdOps::Exp<Z,Z>>(result, resultShapeBuffer, result, resultShapeBuffer, extraParams, tadShapeInfo, tadOffsets);

				//take the sum for the exponential
				functions::reduce::ReduceSameFunction<X>::template exec<simdOps::Sum<X>>(result, resultShapeBuffer, extraParams, maxResult.data(), maxResultShapeBuffer, maxDimension, 1, nullptr, nullptr);

				//divide by the sum
				functions::broadcast::Broadcast<Z,X,Z>::template exec<simdOps::Divide<Z,X,Z>>(result, resultShapeBuffer, maxResult.data(), maxResultShapeBuffer, result, resultShapeBuffer, dimension, 1, nullptr, nullptr, nullptr, nullptr);

				if (resultEleStide >= 1) {
					if (resultEleStide == 1) {
#pragma omp simd
						for (int i = 0; i < len; i++) {
							result[i] = result[i] * (static_cast<X>(1.0f) - result[i]);
						}

					}
					else {
#pragma omp simd
						for (int i = 0; i < len; i++) {
							result[i * resultEleStide] = result[i * resultEleStide] * (static_cast<X>(1.0f) - result[i * resultEleStide]);
						}

					}
				}
				else {
                    auto zShape = shape::shapeOf(resultShapeBuffer);
                    auto zStride = shape::stride(resultShapeBuffer);
                    auto zRank = shape::rank(resultShapeBuffer);

                    Nd4jLong zCoord[MAX_RANK];

                    for (int i = 0; i < len; i++) {
                        shape::ind2subC(zRank,zShape, i, zCoord);
                        Nd4jLong zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);
                        result[zOffset] = result[zOffset] * ((X) 1.0f - result[zOffset]);
                    }
                }


				delete[] maxResultShapeBuffer;
			}
			else if (shape::isVector(xShapeBuffer, 2)) {
				auto max = -nd4j::DataTypeUtils::max<X>();
				X sum = 0;

				auto elementWiseStride = shape::elementWiseStride(xShapeBuffer);
				auto length = shape::length(xShapeBuffer);
				if (elementWiseStride == 1) {

#pragma omp simd reduction(maxT:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<X>(max, result[i]);
					}

#pragma omp simd reduction(sumT:sum)
					for (int i = 0; i < length; i++) {
						result[i] -= max;
						result[i] = nd4j::math::nd4j_exp<X, X>(result[i]);
						sum += result[i];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i] /= sum;
					}

#pragma omp simd
                    for (int i = 0; i < length; i++) {
                        result[i] = result[i] * ((X) 1.0f - result[i]);
                    }
                } else if (elementWiseStride >= 1) {

#pragma omp simd reduction(maxT:max)
					for (int i = 0; i < length; i++) {
						max = nd4j::math::nd4j_max<X>(max, result[i * elementWiseStride]);
					}


#pragma omp simd reduction(sumT:sum)
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] -= max;
						result[i * elementWiseStride] = nd4j::math::nd4j_exp<X, X>(result[i * elementWiseStride]);
						sum += result[i * elementWiseStride];
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] /= sum;
					}

#pragma omp simd
					for (int i = 0; i < length; i++) {
						result[i * elementWiseStride] = result[i * elementWiseStride] * ((X) 1.0f - result[i * elementWiseStride]);
					}
				} else {
                    printf("non-ews access on row not implemented yet");
                }
			}
		}

		op_def static X op(X d1, Z *params) {
			return nd4j::math::softplus<X>(d1);
		}
	};


	template<typename X, typename Z>
	class IsMax {
	public:
		static const bool requiresSpecial = true;


#ifdef __CUDACC__

		static inline  __device__ void doAllCuda(
			T *dx,
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
// this code is safe to delete, it's never used
/*
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
			*/
		}
#endif

#ifdef __CUDACC__
		inline __host__

#elif defined(__GNUC__)


#endif
		static void doAll(
			X *dx,
			Nd4jLong *xShapeBuffer,
            Z *result,
			Nd4jLong *resultShapeBuffer,
			X *extraParams) {

			auto length = shape::length(xShapeBuffer);
			auto eleStride = shape::elementWiseStride(xShapeBuffer);
			auto resultEleStride = shape::elementWiseStride(resultShapeBuffer);
			auto xOrder = shape::order(xShapeBuffer);
			auto resultOrder = shape::order(resultShapeBuffer);
/*
			int tadsPerThread = tads / TAD_THRESHOLD;
			int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
			num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());
*/
			if (xOrder == resultOrder && xOrder == 'c') {
				if (eleStride == 1 && resultEleStride == 1) {
					if (length < ELEMENT_THRESHOLD) {
						int maxIdx = 0;
                        auto currMax = dx[0];
//#pragma omp simd reduction (max:maxIdx,currMax)
						for (int i = 0; i < length; i++) {
							if (currMax < dx[i]) {
								currMax = dx[i];
								maxIdx = i;
							}

							result[i] = false;

						}

						result[maxIdx] = true;

					}
					else {
						int maxIdx = 0;
						auto currMax = dx[0];

#pragma omp parallel proc_bind(AFFINITY)
{
						int maxIdxLocal = maxIdx;
						auto currMaxLocal = currMax;

//#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
						for (int i = 0; i < length; i++) {
							if (currMaxLocal < dx[i]) {
								currMaxLocal = dx[i];
								maxIdxLocal = i;
							}
							result[i] = false;
						}
#pragma omp critical
{
						if (currMax < currMaxLocal) {
							currMax = currMaxLocal;
							maxIdx = maxIdxLocal;
						}
}
}
						result[maxIdx] = true;
					}

				}
				else {
					if (length < ELEMENT_THRESHOLD) {
						int maxIdx = 0;
                        auto currMax = dx[0];
//#pragma omp simd reduction(max:maxIdx,currMax)
						for (int i = 0; i < length; i++) {
							result[i * resultEleStride] = false;
							if (currMax < dx[i * eleStride]) {
								currMax = dx[i * eleStride];
								maxIdx = i;
							}
						}

						result[maxIdx * resultEleStride] = true;

					}
					else {
						int maxIdx = 0;
						auto currMax = dx[0];

#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
						int maxIdxLocal = maxIdx;
						auto currMaxLocal = currMax;
//#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
						for (int i = 0; i < length; i++) {
							result[i * resultEleStride] = false;
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
						result[maxIdx * resultEleStride] = true;
					}

				}
			}


			else {
				Nd4jLong shapeIter[MAX_RANK];
				Nd4jLong coord[MAX_RANK];
				int dim;
				Nd4jLong xStridesIter[MAX_RANK];
				Nd4jLong resultStridesIter[MAX_RANK];
				auto xShape = shape::shapeOf(xShapeBuffer);
				auto xStride = shape::stride(xShapeBuffer);
				auto resultStride = shape::stride(resultShapeBuffer);
				auto rank = shape::rank(xShapeBuffer);
				auto originalResult = result;
				if (PrepareTwoRawArrayIter<X>(rank,
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
					auto value = dx[0];
					int idx = 0;
					int maxIdx = 0;
					ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
						if (dx[0] > value) {
							value = dx[0];
							maxIdx = idx;
						}

						idx++;
						result[0] = (Z) 0;

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
						originalResult[maxIdx] = (Z)1;
					else
						originalResult[maxIdx * shape::stride(resultShapeBuffer)[shape::rank(resultShapeBuffer) - 1]] = (Z)1;
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
			Nd4jLong *xShapeBuffer,
			T *result,
			Nd4jLong *resultShapeBuffer,
			T *extraParams, int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
			// FIXME: MAX_DIMENSION is lower then FP16 frame
			if (extraParams == nullptr || (int) extraParams[0] == MAX_DIMENSION) {
				doAllCuda(dx, xShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, reductionPointer, manager);
			}
		}
#endif

		static void execSpecial(
			X *dx,
			Nd4jLong *xShapeBuffer,
			Z *result,
			Nd4jLong *resultShapeBuffer,
			X *extraParams,
			Nd4jLong *tadShapeInfo,
			Nd4jLong *tadOffsets) {
			//FIXME: this op should be moved to CustomOps
			if (extraParams == nullptr || (int)extraParams[0] == 0 ||
				((int)extraParams[0] == 1 && (int)extraParams[1] == MAX_DIMENSION)) {
				doAll(dx, xShapeBuffer, result, resultShapeBuffer, extraParams);
			}
			else if (shape::isVector(xShapeBuffer)) {
				auto dimensionLength = (int)extraParams[0];
				auto dimension = new int[dimensionLength];
				auto length = shape::length(xShapeBuffer);
				for (int i = 0; i < dimensionLength; i++) {
					dimension[i] = (int)extraParams[i + 1];
				}
				if (shape::shapeOf(xShapeBuffer)[dimension[0]] == 1) {
					for (int i = 0; i < length; i++) {
						result[i] = 1.0;
					}
				}
				else {
					auto eleStride = shape::elementWiseStride(xShapeBuffer);
					if (eleStride == 1) {
						int maxIdx = 0;
						auto currMax = dx[0];
						if (length < ELEMENT_THRESHOLD) {

//#pragma omp simd reduction(max:maxIdx,currMax)
							for (int i = 0; i < length; i++) {
								if (currMax < dx[i]) {
									currMax = dx[i];
									maxIdx = i;
								}

								result[i] = 0.0;

							}
						}
						else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
							int maxIdxLocal = maxIdx;
							auto currMaxLocal = currMax;
//#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
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
						auto currMax = dx[0];
						if (length < ELEMENT_THRESHOLD) {
//#pragma omp parallel for reduction(max:maxIdx,currMax) proc_bind(AFFINITY)
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
							auto currMaxLocal = currMax;

//#pragma omp parallel for reduction(max:maxIdx,currMax)  proc_bind(AFFINITY)
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
                auto dimensionLength = (int) extraParams[0];
                auto dimension = new int[dimensionLength];

#pragma omp simd
                for (int i = 0; i < dimensionLength; i++) {
                    dimension[i] = (int) extraParams[i + 1];
                }
                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem				
				auto tadShapeShapeInfo = tadShapeInfo;
				shape::TAD tad (xShapeBuffer, dimension, dimensionLength);
				if(tadShapeInfo==nullptr) {
					tad.createTadOnlyShapeInfo();
					tad.createOffsets();
					tadShapeShapeInfo = tad.tadOnlyShapeInfo;
					tadOffsets = tad.tadOffsets;
				}						                                				

                auto tadLength = shape::tadLength(xShapeBuffer, dimension, dimensionLength);
                auto tads = shape::length(xShapeBuffer) / tadLength;

                int tadsPerThread = tads / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                auto tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                auto zEWS = tadEWS;

                int span = (tads / num_threads) + 8;

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY)
                {
                    int tid = omp_get_thread_num();
                    int start = span * tid;
                    int end = span * (tid + 1);
                    if (end > tads) end = tads;

                    for (int r = start; r < end; r++) {
                        if (tadEWS > 0 && zEWS > 0 && dimensionLength == 1) {
                            auto rX = dx + tadOffsets[r];
                            auto rZ = result + tadOffsets[r];

                            auto maxValue = rX[0];
                            int maxIdx = 0;
                            if (tadEWS == 1 && zEWS == 1) {
//#pragma omp simd reduction(max:maxValue,maxIdx)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i] = maxIdx == i ? (X) 1.0f : (X) 0.0f;
                                }

                            } else {

//#pragma omp parallel for reduction(max:maxValue,maxIdx) default(shared)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i * tadEWS] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i * tadEWS];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i * zEWS] = maxIdx == i ? (X) 1.0f : (X) 0.0f;
                                }
                            }
                        } else {
                            int tadsPerThread = tads / TAD_THRESHOLD;
                            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                            auto offset = tadOffsets[r];
                            Nd4jLong shapeIter[MAX_RANK];
                            Nd4jLong coord[MAX_RANK];
                            int dim;
                            Nd4jLong xStridesIter[MAX_RANK];
                            Nd4jLong resultStridesIter[MAX_RANK];
                            auto xShape = shape::shapeOf(tadShapeShapeInfo);
                            auto xStride = shape::stride(tadShapeShapeInfo);
                            auto resultStride = shape::stride(tadShapeShapeInfo);
                            int rank = shape::rank(tadShapeShapeInfo);
                            auto xPointer = dx + offset;
                            auto resultPointer = result + offset;
                            auto maxValue = xPointer[0];

                            auto maxCursor = resultPointer;
                            Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                            if (PrepareTwoRawArrayIter<X>(rank,
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
                                   maxCursor = reinterpret_cast<X *>(maxCursorLong);
                                   maxCursor[0] = 1.0;
                            }
                        }
                    }
                }

                delete[] dimension;
            }
		}

		op_def static Z op(X d1, X *params) {
			return nd4j::math::softplus<X>(d1);
		}
	};
}
