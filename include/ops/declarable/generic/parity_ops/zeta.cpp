//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_zeta)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/zeta.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(zeta, 2, 1, false, 0, 0) {

            NDArray<T>* x = INPUT_VARIABLE(0);
            NDArray<T>* q = INPUT_VARIABLE(1);

            NDArray<T>* output   = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(q), 0, "ZETA op: two input arrays must have the same shapes, bot got x=%s and q=%s !", ShapeUtils<T>::shapeAsString(x).c_str(), ShapeUtils<T>::shapeAsString(q).c_str());

            int arrLen = x->lengthOf();

            for(int i = 0; i < arrLen; ++i ) {

                REQUIRE_TRUE((*x)(i) > (T)1., 0, "ZETA op: all elements of x array must be > 1 !");

                REQUIRE_TRUE((*q)(i) > (T)0., 0, "ZETA op: all elements of q array must be > 0 !");
            }

            *output = helpers::zeta<T>(*x, *q);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Zeta, zeta);
    }
}

#endif