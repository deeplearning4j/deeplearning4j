//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        // FIXME: this op badly needs perf improvements!
        CUSTOM_OP_IMPL(unique, 1, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            std::vector<T> values;
            std::vector<int> indices;

            for (int e = 0; e < x->lengthOf(); e++) {
                T v = x->getScalar(e);
                if (std::find(values.begin(), values.end(), v) == values.end()) {
                    values.emplace_back(v);
                    indices.emplace_back(e);
                }
            }


            auto vec_uniq = new NDArray<T>('c', {1, (int) values.size()});
            auto vec_idx = new NDArray<T>('c', {1, (int) values.size()});

            for (int e = 0; e < vec_uniq->lengthOf(); e++) {
                vec_uniq->putScalar(e, values.at(e));
                vec_idx->putScalar(e, indices.at(e));
            }

            OVERWRITE_2_RESULTS(vec_uniq, vec_idx);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(unique) {
            auto shapeList = new ShapeList(); 
            for (int e = 0; e < 2; e++) {
                int* newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
                if (shape::order(inputShape->at(0)) == 'c')
                    shape::shapeBuffer(shape::rank(inputShape->at(0)), shape::shapeOf(inputShape->at(0)), newshape);
                else 
                    shape::shapeBufferFortran(shape::rank(inputShape->at(0)), shape::shapeOf(inputShape->at(0)), newshape);
                shapeList->push_back(newshape); 

            }
            return shapeList;
        }
    }
}