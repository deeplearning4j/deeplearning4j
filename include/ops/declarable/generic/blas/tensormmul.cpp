//
//  @author raver119@gmail.com
//

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        /**
         * tensorMmul/tensorDot operation
         * takes 2 ndarrays, and 2 sets of axes
         *
         * Integer argumens map:
         * IArgs[0] - number of axes along for first array
         * IArgs[1]... axes values for first array
         * IArgs[] - number of axes along for second array
         * IArgs[1]... axes values for second array
         */
        CUSTOM_OP_IMPL(tensormmul, 2, 1, false, 0, -1) {
            NDArray<T>* a = INPUT_VARIABLE(0);
            NDArray<T>* b = INPUT_VARIABLE(1);

            NDArray<T>* c = OUTPUT_VARIABLE(0);                // 

            // building axes
            int axe0_size = INT_ARG(0);
            int axe1_size = INT_ARG(axe0_size+1);
            std::vector<int> axes_0, axes_1;
            for (int e = 0; e < axe0_size; e++)
                axes_0.emplace_back((int) INT_ARG(e+1));

            for (int e = 0; e < axe1_size; e++)
                axes_1.emplace_back((int) INT_ARG(e + axe0_size + 2));

            nd4j_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());

            nd4j::NDArrayFactory<T>::tensorDot(a, b, c, axes_0, axes_1);

            STORE_RESULT(*c);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(tensordot, tensormmul);


        DECLARE_SHAPE_FN(tensormmul) {               
        
            NDArray<T> *a = INPUT_VARIABLE(0);
            NDArray<T> *b = INPUT_VARIABLE(1);  
            // building axes
            int axe0_size = INT_ARG(0);
            int axe1_size = INT_ARG(axe0_size+1);
            std::vector<int> axes_0, axes_1;
            for (int e = 0; e < axe0_size; e++)
                axes_0.emplace_back((int) INT_ARG(e+1));
            for (int e = 0; e < axe1_size; e++)
                axes_1.emplace_back((int) INT_ARG(e + axe0_size + 2));

            // evaluate shapes 
            std::vector<int> permutAt, permutBt, shapeAt, shapeBt;
            std::vector<int> outShape = nd4j::ShapeUtils<T>::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);
            
            int rank = outShape.size();

            int* newShapeInfo = nullptr; 
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int); 
            newShapeInfo[0] = rank;
            copy(outShape.begin(), outShape.end(), newShapeInfo+1);
            shape::updateStrides(newShapeInfo, 'c');

            return new ShapeList(newShapeInfo);
        }
    }
}