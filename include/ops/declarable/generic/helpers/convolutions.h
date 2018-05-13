//
// Based on PyTorch - https://github.com/pytorch/pytorch
//

#ifndef LIBND4J_CONVOLUTIONS_H
#define LIBND4J_CONVOLUTIONS_H

#include <NDArray.h>

namespace nd4j {
    namespace ops {

        template <typename T>
        class ConvolutionUtils {
        public:
            static Nd4jIndex convsize(Nd4jIndex x, Nd4jIndex k, Nd4jIndex s, const char* vf);

            static Nd4jStatus conv3D(T* output_data, T alpha, T* ptr_input, Nd4jIndex nInputDepth, Nd4jIndex nInputRows, Nd4jIndex nInputCols, T* ptr_weight, Nd4jIndex nKernelDepth, Nd4jIndex nKernelRows, Nd4jIndex nKernelCols, Nd4jIndex sdepth, Nd4jIndex srow, Nd4jIndex scol, const char *vf, const char *xc);

            static Nd4jStatus conv3Dmv(NDArray<T>* r_, T beta, T alpha, NDArray<T>* t_, NDArray<T>* k_, Nd4jIndex sdepth, Nd4jIndex srow, Nd4jIndex scol, const char *vf, const char *xc);

            static void fullXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc);

            static void fullConv3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc);

            static void validXCorr3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc);

            static void validConv3Dptr(T*r_, T alpha, T *t_, Nd4jIndex it, Nd4jIndex ir, Nd4jIndex ic, T *k_, Nd4jIndex kt, Nd4jIndex kr, Nd4jIndex kc, Nd4jIndex st, Nd4jIndex sr, Nd4jIndex sc);

            static void _dilatedMaxPool3D(T *input_p, T *output_p, T *indz_p, Nd4jIndex nslices, Nd4jIndex itime, Nd4jIndex iwidth, Nd4jIndex iheight, Nd4jIndex otime, Nd4jIndex owidth, Nd4jIndex oheight, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH);

            static void _dilatedMaxPool3D_bp(T *gradInput_p, T *gradOutput_p, T *indz_p, Nd4jIndex nslices, Nd4jIndex  itime, Nd4jIndex  iwidth, Nd4jIndex  iheight, Nd4jIndex otime, Nd4jIndex owidth, Nd4jIndex oheight, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH);

            static void avgPool3D(NDArray<T>& input, NDArray<T>& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad);

            static void avgPool3DBP(NDArray<T>& gradO, NDArray<T>& gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const bool count_include_pad);
            
            // volume [bS, volC, volD, volH, volW], columns [bS, volC, kD, kH, kW, colD, colH, colW]
            static void vol2col2(NDArray<T>& vol, NDArray<T>& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            // col [bS, volC, kD, kH, kW, colD, colH, colW], vol [bS, volC, volD, volH, volW]
            static void col2vol2(NDArray<T>& col, NDArray<T>& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);            

            static void calcOutSizePool2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode);

            static void calcOutSizePool3D(int& oD, int& oH, int& oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW, const int iD, const int iH, const int iW, const int isSameMode);

            static void _calcPadding2D(int& pH, int& pW, int oH, int oW, int inH, int inW, int kH, int kW, int sH, int sW, int dH, int dW);

            static void calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW);            

            // input [bS, iC, iD, iH, iW], output [bS, iC, oD, oH, oW]
            static void maxPool3d(NDArray<T>& input, NDArray<T>& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            // input [bS, iC, iD, iH, iW], indices [bS, iC, oD, oH, oW]
            static void maxPool3dIndices(NDArray<T>& input, int* indices, const int oD, const int oH, const int oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            // input [bS, iC, iD, iH, iW], indices [bS, iC, iD, iH, iW], output [bS, iC, oD, oH, oW]
            static void maxPool3dBP(NDArray<T>& input, const int* indices, NDArray<T>& output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            // calculation of output height and width in 2D deconvolution procedure
            static void calcOutSizeDeconv2D(int& oH, int& oW, const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW, const int iH, const int iW, const int isSameMode);

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static void getSizesAndIndexesConv2d(const bool isNCHW, const NDArray<T>& input, const NDArray<T>& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH);
            static void getSizesAndIndexesConv2d(const bool isNCHW, const int* inShapeInfo, const int* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH);

            // evaluates sizes values and indexes using input and output arrays depending on data format
            static void getSizesAndIndexesConv3d(const bool isNCDHW, const NDArray<T>& input, const NDArray<T>& output, int& bS, int& iC, int& iD, int& iH, int& iW, int& oC, int& oD, int& oH, int& oW, int& indIOioC, int& indIOioD, int& indWiC, int& indWoC, int& indWkD);

            static void conv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs);

            static void conv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs);

            static void depthwiseConv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs);

            static void depthwiseConv2dBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<int>& intArgs);

            static void sconv2d(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* output, const std::vector<int>& intArgs);

            static void vol2col(NDArray<T>& vol, NDArray<T>& col, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void col2vol(NDArray<T>& col, NDArray<T>& vol, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int dD, const int dH, const int dW);

            static void upsampling2d(const NDArray<T>& input, NDArray<T>& output, const int factorH, const int factorW, const bool isNCHW);

            static void upsampling3d(const NDArray<T>& input, NDArray<T>& output, const int factorD, const int factorH, const int factorW, const bool isNCDHW);

            static void upsampling2dBP(const NDArray<T>& gradO, NDArray<T>& gradI, const bool isNCHW);

            static void upsampling3dBP(const NDArray<T>& gradO, NDArray<T>& gradI, const bool isNCDHW);

            static void maxPool2d(NDArray<T>* input, NDArray<T>* values, const std::vector<int>& params, NDArray<T>* indices);

    };

}
}
#endif //LIBND4J_CONVOLUTIONS_H
