//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/dilation2d.h>
#include <array/DataTypeUtils.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void _dilation2d(NDArray<T> *input, NDArray<T> *weights, NDArray<T> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {
        const int batch = input->sizeAt(0);
        const int input_rows = input->sizeAt(1);
        const int input_cols = input->sizeAt(2);
        const int depth = input->sizeAt(3);

        const int filter_rows = weights->sizeAt(0);
        const int filter_cols = weights->sizeAt(1);

        const int output_rows = output->sizeAt(1);
        const int output_cols = output->sizeAt(2);

#pragma omp parallel for simd schedule(guided)
        for (int b = 0; b < batch; ++b) {
            for (int h_out = 0; h_out < output_rows; ++h_out) {
                int h_beg = h_out * stride_rows - pad_top;
                for (int w_out = 0; w_out < output_cols; ++w_out) {
                    int w_beg = w_out * stride_cols - pad_left;
                    for (int d = 0; d < depth; ++d) {
                        T cur_val = -DataTypeUtils::max<T>();
                        for (int h = 0; h < filter_rows; ++h) {
                            const int h_in = h_beg + h * rate_rows;
                            if (h_in >= 0 && h_in < input_rows) {
                                for (int w = 0; w < filter_cols; ++w) {
                                    const int w_in = w_beg + w * rate_cols;
                                    if (w_in >= 0 && w_in < input_cols) {
                                        const T val = (*input)(b, h_in, w_in, d) + (*weights)(h, w, d);
                                        if (val > cur_val) {
                                            cur_val = val;
                                        }
                                    }
                                }
                            }
                        }
                        (*output)(b, h_out, w_out, d) = cur_val;
                    }
                }
            }
        }
    };


    template void _dilation2d<float>(NDArray<float> *input, NDArray<float> *weights, NDArray<float> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);
    template void _dilation2d<float16>(NDArray<float16> *input, NDArray<float16> *weights, NDArray<float16> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);
    template void _dilation2d<double>(NDArray<double> *input, NDArray<double> *weights, NDArray<double> *output, int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left);
}
}
}