//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_select)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(select, 3, 1, false, 0, 0) {
            auto cond = INPUT_VARIABLE(0);
            auto x = INPUT_VARIABLE(1);
            auto y = INPUT_VARIABLE(2);

            REQUIRE_TRUE(x->isSameShape(y), 0, "Select: X and Y shape should be equal");
            if (x->isScalar()) {
                REQUIRE_TRUE(cond->isScalar(), 0, "Select: Condition should gave either equal shape to X/Y first dimension or to be scalar");

                auto z = OUTPUT_VARIABLE(0);

                T v = cond->getIndexedScalar(0)  == (T) 0.0f ? y->getIndexedScalar(0) : x->getIndexedScalar(0);

                z->putIndexedScalar(0, v);
            } else {
                bool same = cond->isSameShape(x);
                REQUIRE_TRUE(cond->isScalar() || cond->lengthOf() == x->sizeAt(0) || same, 0, "Select: Condition should gave either equal shape to X/Y first dimension or to be scalar");
                if (same) {
                    auto z = OUTPUT_VARIABLE(0);

                    for (int e = 0; e < cond->lengthOf(); e++) {
                        T v = cond->getIndexedScalar(e);
                        T r = v == (T) 0.0f ? y->getIndexedScalar(e) : x->getIndexedScalar(e);
                        z->putIndexedScalar(e, r);
                    }
                } else {
                    REQUIRE_TRUE(cond->lengthOf() == x->sizeAt(0), 0, "Condition length should be equal to the dim0 of x/y to act as TAD-mask, but got %d instead", cond->lengthOf());

                    auto z = OUTPUT_VARIABLE(0);

                    auto dims = ShapeUtils<T>::convertAxisToTadTarget(x->rankOf(), {0});
                    auto tadsX = x->allTensorsAlongDimension(dims);
                    auto tadsY = y->allTensorsAlongDimension(dims);
                    auto tadsZ = z->allTensorsAlongDimension(dims);

                    for (int e = 0; e < tadsX->size(); e++) {
                        T v = cond->getIndexedScalar(e);
                        
                        if (v == (T) 0.0f)
                            tadsZ->at(e)->assign(tadsY->at(e));
                        else
                            tadsZ->at(e)->assign(tadsX->at(e));
                    }

                    delete tadsX;
                    delete tadsY;
                    delete tadsZ;
                }
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(select) {
            auto inShape = inputShape->at(1);

            Nd4jLong *newshape;
            COPY_SHAPE(inShape, newshape);

            return SHAPELIST(newshape);
        }
    }
}

#endif