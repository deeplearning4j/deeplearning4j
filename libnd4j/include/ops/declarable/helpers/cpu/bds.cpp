//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/bds.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int bdsFunctor(NDArray<T>* x_shape, NDArray<T>* y_shape, NDArray<T>* output) {
        int e = 0, x = 0, y = 0;
//#pragma omp parallel for
        for ( ; e < output->lengthOf(); e++) {
            T val;
            if (x < x_shape->lengthOf() && y < y_shape->lengthOf()) {
                val = nd4j::math::nd4j_max((*x_shape)(x++), (*y_shape)(y++));
            }
            else if (x < x_shape->lengthOf()) {
                val = nd4j::math::nd4j_max((*x_shape)(x++), (*y_shape)(y - 1));
            }
            else if (y < y_shape->lengthOf()) {
                val = nd4j::math::nd4j_max((*x_shape)(x - 1), (*y_shape)(y++));
            }
            else {
                //REQUIRE_TRUE(e < 0, 0, "broadcast_dynamic_shape: Wrong value in a shape vector");
                return ND4J_STATUS_OK;
            }
            if (e)
                if (val != (*output)(e - 1)) {
                    nd4j_printf("broadcast_dynamic_shape: Input shapes should be compatible", "");
                    return ND4J_STATUS_VALIDATION;
                }
            (*output)(e) = val;
        }
        return ND4J_STATUS_OK;
    }

    template int bdsFunctor(NDArray<float>* x_shape, NDArray<float>* y_shape, NDArray<float>* output);
    template int bdsFunctor(NDArray<float16>* x_shape, NDArray<float16>* y_shape, NDArray<float16>* output);
    template int bdsFunctor(NDArray<double>* x_shape, NDArray<double>* y_shape, NDArray<double>* output);
}
}
}