//
//  @author Adam Gibson
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_choose)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/ops.h>
#include <vector>
#include <NDArray.h>

template<typename T>
nd4j::NDArray<T>  * processCondition(int mode,nd4j::NDArray<T> *arg, nd4j::NDArray<T> *comp, T compScalar);

template <typename T>
T processElementCondition(int mode,T d1,T d2);




template<typename T>
nd4j::NDArray<T>  * processCondition(int mode,nd4j::NDArray<T> *arg, nd4j::NDArray<T> *comp,nd4j::NDArray<T> *output, nd4j::NDArray<T> *numResult,T compScalar) {
    /**
     * Convert to straight ndarray based on input
     */
    int numResults = 0;
    if(comp != nullptr) {
        if (comp->isScalar()) {
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            nd4j::NDArray<T> arg1 = *arg;
            nd4j::NDArray<T> comp1 = *comp;
            for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition<T>(mode,arg1(i),comp1(0));
                if(result2 > 0) {
                    output->putScalar(numResults, arg1(i));
                    numResults++;
                }
            }
        } else {
            // REQUIRE_TRUE(comp.isSameShape(arg));
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            nd4j::NDArray<T> arg1 = *arg;
            for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition<T>(mode,arg1(i),compScalar);
                if(result2 > 0) {
                    output->putScalar(numResults, arg1(i));
                    numResults++;
                }
            }
        }

    }
    else {
        nd4j::NDArray<T> arg1 = *arg;
        //Other input for compare could be an ndarray or a secondary scalar
        //for comparison
        for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
            T result2 = processElementCondition<T>(mode,arg1(i),compScalar);
            if(result2 > 0) {
                output->putScalar(numResults, arg1(i));
                numResults++;
            }
        }
    }

    if(numResult != nullptr)
        numResult->putScalar(0,numResults);

    return output;

}

template <typename T>
T processElementCondition(int mode,T d1,T d2) {
    T modePointer = (T ) mode;
    T input[3] = {d2, (T) EPS, (T) mode};
    T res = simdOps::MatchCondition<T>::op(d1, input);
    return res;

}


namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(choose, -1, 2, false, -1, -1) {
            int mode = INT_ARG(0);
            if (block.width() > 1) {
                auto arg = INPUT_VARIABLE(0);
                auto comp = INPUT_VARIABLE(1);
                auto result = OUTPUT_VARIABLE(0);
                auto numResults = OUTPUT_VARIABLE(1);
                auto  arg1 = *arg;
                auto comp1 = *comp;
                if(arg->isScalar() || comp->isScalar()) {
                    if(arg->isScalar()) {
                        T scalar = arg1(0);
                        processCondition<T>(mode,comp,nullptr,result,numResults,scalar);

                    }
                    else {
                        T scalar = comp1(0);
                        processCondition<T>(mode,arg,nullptr,result,numResults,scalar);

                    }
                }
                else {
                    processCondition<T>(mode,arg,comp,result,numResults,0.0f);

                }



                STORE_2_RESULTS(result,numResults);

            }//scalar case
            else {
                T scalar = (T) T_ARG(0);
                auto arg = INPUT_VARIABLE(0);
                auto numResults = OUTPUT_VARIABLE(1);
                auto result = OUTPUT_VARIABLE(0);
                processCondition<T>(mode,arg,nullptr,result,numResults,scalar);
                STORE_2_RESULTS(result,numResults);
            }


            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(choose) {
            Nd4jLong *shape;
            int rank;
            if(block.width() > 1) {
                auto first = INPUT_VARIABLE(0);
                auto second = INPUT_VARIABLE(1);
                if(first->lengthOf() > second->lengthOf()) {
                    shape = first->getShapeInfo();
                    rank = first->rankOf();
                }
                else {
                    shape = second->getShapeInfo();
                    rank = second->rankOf();
                }
            }
            else {
                auto first = INPUT_VARIABLE(0);
                shape = first->getShapeInfo();
                rank = first->rankOf();
            }

            Nd4jLong* newShape;
            COPY_SHAPE(shape, newShape);

            auto shapeScalar = ShapeUtils<T>::createScalarShapeInfo(block.workspace());

            return SHAPELIST(newShape, shapeScalar);
        }


    }
}

#endif