//
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        // here iArgs is a vector with (optional) negative of order as first element:
        // ({-order, dim1, dim2, dim3, ...})
        CUSTOM_OP_IMPL(reshape, 1, 1, true, 0, -2) {
            auto x = INPUT_VARIABLE(0);

            if (block.width() == 1) {
                auto arguments = block.getIArguments();
                int argsSize = arguments->size();

                int e = 1;
                char order = (char) -(*arguments)[0];
                if (order != 'c' && order != 'f') {
                    order = x->ordering();
                    e = 0;
                }

                REQUIRE_TRUE(argsSize - e >= 1, 0, "Reshape arguments should at least 1 dimension");

                std::vector<Nd4jLong> shapeNew;
                int e2 = e;
                for (; e < (int) arguments->size(); e++) {
                    if (arguments->at(e) == -1){
                        long shapeLength = 1;
                        for(; e2 < e; e2++){
                            shapeLength *= arguments->at(e2);
                        }
                        for(e2 = e + 1; e2 < arguments->size(); e2++){
                            shapeLength *= arguments->at(e2);
                        }
                        int realShape = nd4j::math::nd4j_abs<int>((int) x->lengthOf() / shapeLength);
                        shapeNew.push_back(realShape);
                    }
                    else{
                        shapeNew.push_back((int) arguments->at(e));
                    }

                }

                auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
                REQUIRE_TRUE(len == x->lengthOf(), 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), len);

                if (Environment::getInstance()->isDebugAndVerbose()) {
                    nd4j_printv("Reshape: new shape", shapeNew);
                }

                if (block.isInplace()) {
                    if (x->reshapei(order, shapeNew)) {
                        STORE_RESULT(*x);
                        return ND4J_STATUS_OK;
                    }
                } else {
                    auto ret = OUTPUT_VARIABLE(0);
                    auto xr = x->reshape(order, shapeNew);
                    ret->assign(xr);
                    STORE_RESULT(*ret);
                    delete xr;
                    return ND4J_STATUS_OK;
                }
            } else if (block.width() == 2) {
                auto s = INPUT_VARIABLE(1);

                char order = 'c';
                if (block.numI() > 0)
                    order = (char) -INT_ARG(0);

                std::vector<Nd4jLong> shapeNew(s->lengthOf());

                for (int e = 0; e < (int) s->lengthOf(); e++) {
                    auto dim = static_cast<Nd4jLong>(s->getScalar(e));
                    if (dim == -1){
                        long shapeLength = 1;
                        for(int e2 = 0; e2 < e; e2++){
                            shapeLength *= static_cast<Nd4jLong>(s->getScalar(e2));
                        }
                        for(int e2 = e + 1; e2 < (int) s->lengthOf(); e2++){
                            REQUIRE_TRUE(static_cast<Nd4jLong>(s->getScalar(e2)) != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                            shapeLength *= static_cast<Nd4jLong>(s->getScalar(e2));
                        }
                        int realShape = nd4j::math::nd4j_abs<int>((int) x->lengthOf() / shapeLength);
                        shapeNew[e] = realShape;
                    }
                    else{
                        shapeNew[e] = dim;
                    }
                }

                if (Environment::getInstance()->isDebugAndVerbose()) {
                    nd4j_printv("Reshape: new shape", shapeNew);
                }

                if (block.isInplace()) {
                    if (x->reshapei(order, shapeNew)) {
                        STORE_RESULT(*x);
                        return Status::OK();
                    }
                } else {
                    auto ret = OUTPUT_VARIABLE(0);
                    if (s->isEmpty()) {
                        // just a scalar
                        ret->assign(x);
                    } else {
                        auto xr = x->reshape(order, shapeNew);
                        ret->assign(xr);
                        delete xr;
                    }

                    return Status::OK();
                }
            }

            return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SHAPE_FN(reshape) {
            auto inp = inputShape->at(0);

            // we can launch op using Int arguments
            if (inputShape->size() == 1) {
                std::vector<int> *arguments = block.getIArguments();

                int e = 1;
                char order = (char) -(*arguments)[0];
                if (order != 'c' && order != 'f') {
                    order = shape::order(inp);
                    e = 0;
                }

                std::vector<int> shapeNew;

                int e2 = e;
                for (; e < (int) arguments->size(); e++) {
                    if ((int) arguments->at(e) == -1){

                        long shapeLength = 1;
                        for(; e2 < e; e2 ++){
                            shapeLength *= arguments->at(e2);
                        }
                        for(e2 = e + 1; e2 < arguments->size(); e2++){
                            REQUIRE_TRUE(arguments->at(e2) != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                            shapeLength *= arguments->at(e2);
                        }

                        int realShape = nd4j::math::nd4j_abs<int>((int) shape::length(inp) / shapeLength);
                        shapeNew.push_back(realShape);
                    }
                    else{
                        shapeNew.push_back((int) arguments->at(e));
                    }
                }

                Nd4jLong *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) shapeNew.size()), Nd4jLong);


                newShape[0] = shapeNew.size();
                int cnt = 1;
                for (auto v: shapeNew)
                    newShape[cnt++] = v;

                shape::updateStrides(newShape, order);

                return SHAPELIST(newShape);
            } else {
                // or, with second input "as shape"
                auto x = INPUT_VARIABLE(0);
                auto y = INPUT_VARIABLE(1);

                // special case here
                if (y->isEmpty()) {
                    REQUIRE_TRUE(x->lengthOf() == 1, 0, "Reshape: new length doesn't match existing array");


                    return SHAPELIST(ShapeUtils<T>::createScalarShapeInfo(block.getWorkspace()));
                }

                std::vector<Nd4jLong> shapeNew(y->lengthOf());

                for (int e = 0; e < (int) y->lengthOf(); e++) {
                    auto dim = (int)y->getIndexedScalar(e);
                    if (dim == -1){
                        long shapeLength = 1;
                        for(int e2 = 0; e2 < e; e2++){
                            shapeLength *= (int)y->getIndexedScalar(e2);
                        }
                        for(int e2 = e + 1; e2 < (int)y->lengthOf(); e2++){
                            REQUIRE_TRUE((int)y->getIndexedScalar(e2) != -1, 0, "Reshape : Only one unknown dimension (-1) is allowed.");
                            shapeLength *= (int)y->getIndexedScalar(e2);
                        }
                        int realShape = nd4j::math::nd4j_abs<int>((int) shape::length(inp) / shapeLength);
                        shapeNew[e] = realShape;
                    }else {
                        shapeNew[e] = dim;
                    }
                }


                Nd4jLong *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shapeNew.size()), Nd4jLong);

                shape::shapeBuffer(shapeNew.size(), shapeNew.data(), newShape);

                return SHAPELIST(newShape);
            }
        }
    }
}

#endif