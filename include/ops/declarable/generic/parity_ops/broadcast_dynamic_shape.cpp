//
//  @author raver119@gmail.com
//

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(broadcast_dynamic_shape, 2, 1, false, 0, 0) {

            NDArray<T>* x_shape = INPUT_VARIABLE(0);
            NDArray<T>* y_shape = INPUT_VARIABLE(1);
            
            REQUIRE_TRUE(x_shape->isVector(), 0, "broadcast_dynamic_shape: The first argument should be a vector");
            REQUIRE_TRUE(y_shape->isVector(), 0, "broadcast_dynamic_shape: The second argument should be a vector");

            NDArray<T>* output = OUTPUT_VARIABLE(0);
     
            for (int e = 0, x = 0, y = 0; e < output->lengthOf(); e++) {
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
                    REQUIRE_TRUE(val == (*output)(e - 1), 0, "broadcast_dynamic_shape: Input shapes should be compatible");
                (*output)(e) = val;
            }
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(broadcast_dynamic_shape) {
            auto shapeList = new ShapeList(); 
            
            auto theFirst = inputShape->at(0);
            auto theSecond = inputShape->at(1);

            int theFirstLen = shape::sizeAt(theFirst, -1);
            int theSecondLen = shape::sizeAt(theSecond, -1);

            int* newshape;
    
            int shapeLength = nd4j::math::nd4j_max(theFirstLen, theSecondLen);

            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(1), int);
            shape::shapeVector(shapeLength,  newshape);

            shapeList->push_back(newshape); 
            return shapeList;
        }

    }
}