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
template<typename T, typename AF> class Pooling2DLayer: public BaseLayer<T, AF> {
    protected:
        //  0: max, 1: avg, 2: pnorm
        int _poolingMode;
        int _kernelWidth;
        int _kernelHeight;
        int _strideX;
        int _strideY;
        int _padWidth;
        int _padHeight;
        int _heightCol;
        int _widthCol;
    
        T _extraParam0;
        T _extraParam1;
        T _extraParam2;
    
        int *_im2colShape;
    
    public:
        
        // default constructor 
        Pooling2DLayer();
        
        int configurePooling2D(const int poolingMode, const int kernelHeight, const int kernelWidth, const int strideHeight, const int strideWidth, const int padHeight, const int padWidth, const int outH, const int outW);
    
        virtual int feedForward(); 
    
        virtual int backPropagate() {
            // to be implemented    
            return ND4J_STATUS_OK;
        }
};



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

template<typename T, typename AF> Pooling2DLayer<T,AF>::Pooling2DLayer() {
    
    _poolingMode = 0;
    _kernelWidth = 0;
    _kernelHeight= 0;
    _strideX     = 0;
    _strideY     = 0;
    _padWidth    = 0;
    _padHeight   = 0;
    _heightCol   = 0;
    _widthCol    = 0;
    _extraParam0 = (T) 0.f;
    _extraParam1 = (T) 0.f;
    _extraParam2 = (T) 0.f;
    _im2colShape = nullptr;
}



template<typename T, typename AF> 
int Pooling2DLayer<T,AF>::configurePooling2D(const int poolingMode, const int kernelHeight, const int kernelWidth, const int strideHeight, const int strideWidth, const int padHeight, const int padWidth, const int outH, const int outW) {

    _poolingMode = poolingMode;
    _kernelHeight = kernelHeight;
    _kernelWidth = kernelWidth;
    _strideX = strideWidth;
    _strideY = strideHeight;
    _padHeight = padHeight;
    _padWidth = padWidth;
    _heightCol = outH;
    _widthCol = outW;
    
    int *shape = new int[6];
    shape[0] = this->inputShapeInfo[0];
    shape[1] = this->inputShapeInfo[1];
    shape[2] = kernelHeight;
    shape[3] = kernelWidth;
    shape[4] = outH;
    shape[5] = outW;
    
    _im2colShape = shape::shapeBuffer(6, shape);
    
    
    delete[] shape;
    
    return ND4J_STATUS_OK;
}



template<typename T, typename AF> int Pooling2DLayer<T,AF>::feedForward() {
    int kSize = _kernelWidth * _kernelHeight;

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

    int *outShape = shape::shapeOf(_im2colShape);
    int *outStride = shape::stride(_im2colShape);

    int n = samples * depth * _heightCol * _widthCol;

    int threads = omp_get_max_threads();
    int span = (n / threads) + 1;



#pragma omp parallel num_threads(threads) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        int start = span * tid;
        int end = span * (tid + 1);
        if (end > n) end = n;
        T res;

        for (int index = start; index < end; index++) {
            int h_index = index / _widthCol;
            int h_col = h_index % _heightCol;
            int w_col = index % _widthCol;

            int c_im = h_index / _heightCol;
            int c_col = c_im * kSize;

            int depth_im = c_im % depth;
            int num_im = c_im / depth;
            int h_offset = h_col * _strideY - _padHeight;
            int w_offset = w_col * _strideX - _padWidth;

            T *data_col_ptr = this->_output;

            int i_c = (c_col * _heightCol + h_col) * _widthCol + w_col;
            data_col_ptr += (c_col * _heightCol + h_col) * _widthCol + w_col;

            T *data_im_ptr = this->_input;

            data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset * stridew;
            res = _poolingMode == 0 ? (T) -MAX_FLOAT : (T) 0.0f;

            for (int i = 0; i < _kernelHeight; ++i) {
                for (int j = 0; j < _kernelWidth; ++j) {
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
                    if (_poolingMode == 0) {
                        if (res < val)
                            res = val;
                        // avg
                    } else if (_poolingMode == 1) {
                        res += val;

                        // phorm
                    } else if (_poolingMode == 2) {
                        res += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(val), _extraParam0);
                    }

                    //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                    data_col_ptr += _heightCol * _widthCol;
                    i_c += _heightCol * _widthCol;
                }
            }

            // avg final step
            if (_poolingMode == 1) {
                res /= kSize;

                // pnorm final step
            } else if (_poolingMode == 2) {
                res = nd4j::math::nd4j_pow<T>(res, (T) 1.0f /  _extraParam0);
            }

            this->_output[index] = res;
        }
    }

    return ND4J_STATUS_OK;
}


// end of namespace brackets
}
}

#endif //PROJECT_POOLING2DLAYER_H
