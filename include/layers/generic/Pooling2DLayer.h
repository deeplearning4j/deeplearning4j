//
// @author raver119@gmail.com
//

#ifndef PROJECT_POOLING2DLAYER_H
#define PROJECT_POOLING2DLAYER_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
    namespace layers {

        // FIXME: we don't need activation function here
        template<typename T, typename AF>
        class Pooling2DLayer: public BaseLayer<T, AF> {
        protected:
            /**
             * 0: max
             * 1: avg
             * 2: pnorm
             */
            int poolingMode = 0;
            int kernelWidth = 0;
            int kernelHeight = 0;
            int strideX = 0;
            int strideY = 0;
            int padWidth = 0;
            int padHeight = 0;
            int height_col = 0;
            int width_col = 0;

            T extraParam0;
            T extraParam1;
            T extraParam2;

            int *im2colShape;

        public:

            int configurePooling2D(int poolingMode, int kernelHeight, int kernelWidth, int strideHeight, int strideWidth, int padHeight, int padWidth, int outH, int outW) {
                this->poolingMode = poolingMode;
                this->kernelHeight = kernelHeight;
                this->kernelWidth = kernelWidth;
                this->strideX = strideWidth;
                this->strideY = strideHeight;
                this->padHeight = padHeight;
                this->padWidth = padWidth;
                this->height_col = outH;
                this->width_col = outW;

                int *shape = new int[6];
                shape[0] = this->inputShapeInfo[0];
                shape[1] = this->inputShapeInfo[1];
                shape[2] = kernelHeight;
                shape[3] = kernelWidth;
                shape[4] = outH;
                shape[5] = outW;

                im2colShape = shape::shapeBuffer(6, shape);


                delete[] shape;

                return ND4J_STATUS_OK;
            }


            int feedForward() {
                int kSize = kernelWidth * kernelHeight;

                int *inShape = shape::shapeOf(this->inputShapeInfo);
                int *inStride = shape::stride(this->inputShapeInfo);

                int samples = inShape[0];
                int depth = inShape[1];
                int height = inShape[2];
                int width = inShape[3];


                int strideex = inStride[0];
                int stridech = inStride[1];
                int strideh = inStride[2];
                int stridew = inStride[3];

                int *outShape = shape::shapeOf(im2colShape);
                int *outStride = shape::stride(im2colShape);

                int n = samples * depth * height_col * width_col;

                int _threads = omp_get_max_threads();
                int span = (n / _threads) + 1;



#pragma omp parallel num_threads(_threads) proc_bind(close)
                {
                    int tid = omp_get_thread_num();
                    int start = span * tid;
                    int end = span * (tid + 1);
                    if (end > n) end = n;
                    T res;

                    for (int index = start; index < end; index++) {
                        int h_index = index / width_col;
                        int h_col = h_index % height_col;
                        int w_col = index % width_col;

                        int c_im = h_index / height_col;
                        int c_col = c_im * kSize;

                        int depth_im = c_im % depth;
                        int num_im = c_im / depth;
                        int h_offset = h_col * strideY - padHeight;
                        int w_offset = w_col * strideX - padWidth;

                        T *data_col_ptr = this->output;

                        int i_c = (c_col * height_col + h_col) * width_col + w_col;
                        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                        T *data_im_ptr = this->input;

                        data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset * stridew;
                        res = poolingMode == 0 ? (T) -MAX_FLOAT : (T) 0.0f;

                        for (int i = 0; i < kernelHeight; ++i) {
                            for (int j = 0; j < kernelWidth; ++j) {
                                int h_im = h_offset + i;
                                int w_im = w_offset + j;
                                int i_f = 0;
                                int i_c_temp = i_c;
                                for (int dim = 5; dim >= 0; dim--) {
                                    i_f += (i_c_temp % outShape[dim]) * outStride[dim];
                                    i_c_temp = i_c_temp / outShape[dim];
                                }

                                T val;
                                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                    val = data_im_ptr[i * strideh + j * stridew];
                                else
                                    val = (T) 0.0f;

                                //kernel[i * kernelHeight + j] = val;
                                // max
                                if (poolingMode == 0) {
                                    if (res < val)
                                        res = val;
                                    // avg
                                } else if (poolingMode == 1) {
                                    res += val;

                                    // phorm
                                } else if (poolingMode == 2) {
                                    res += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(val), extraParam0);
                                }

                                //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                                data_col_ptr += height_col * width_col;
                                i_c += height_col * width_col;
                            }
                        }

                        // avg final step
                        if (poolingMode == 1) {
                            res /= kSize;

                            // pnorm final step
                        } else if (poolingMode == 2) {
                            res = nd4j::math::nd4j_pow<T>(res, (T) 1.0f /  extraParam0);
                        }

                        this->output[index] = res;
                    }
                }

                return ND4J_STATUS_OK;
            }

            int backPropagate() {
                // to be implemented

                return ND4J_STATUS_OK;
            }
        };
    }
}

#endif //PROJECT_POOLING2DLAYER_H
