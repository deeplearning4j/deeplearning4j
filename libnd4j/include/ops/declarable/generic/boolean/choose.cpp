/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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

// FIXME: !!!
/*
nd4j::NDArray* processCondition(int mode,nd4j::NDArray *arg, nd4j::NDArray *comp, nd4j::NDArray& compScalar);

template <typename T>
T processElementCondition(int mode,T d1,T d2);



nd4j::NDArray* processCondition(int mode,nd4j::NDArray *arg, nd4j::NDArray *comp,nd4j::NDArray *output, nd4j::NDArray *numResult, nd4j::NDArray& compScalar) {

     //Convert to straight ndarray based on input

    int numResults = 0;
    if(comp != nullptr) {
        if (comp->isScalar()) {
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            nd4j::NDArray arg1 = *arg;
            nd4j::NDArray comp1 = *comp;
            for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition(mode,arg1(i),comp1(0.));
                if(result2 > 0) {
                    output->putScalar(numResults, arg1(i));
                    numResults++;
                }
            }
        } else {
            // REQUIRE_TRUE(comp.isSameShape(arg));
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            nd4j::NDArray arg1 = *arg;
            for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition(mode,arg1(i),compScalar);
                if(result2 > 0) {
                    output->putScalar(numResults, arg1(i));
                    numResults++;
                }
            }
        }

    }
    else {
        nd4j::NDArray arg1 = *arg;
        //Other input for compare could be an ndarray or a secondary scalar
        //for comparison
        for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
            T result2 = processElementCondition(mode,arg1(i),compScalar);
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
    T res = simdOps::MatchCondition::op(d1, input);
    return res;

}
*/

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(choose, -1, 2, false, -1, -1) {
            /*
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
                        T scalar = arg1(0.);
                        processCondition(mode,comp,nullptr,result,numResults,scalar);

                    }
                    else {
                        T scalar = comp1(0.);
                        processCondition(mode,arg,nullptr,result,numResults,scalar);

                    }
                }
                else {
                    processCondition(mode,arg,comp,result,numResults,0.0f);

                }



                STORE_2_RESULTS(result,numResults);

            }//scalar case
            else {
                T scalar = (T) T_ARG(0);
                auto arg = INPUT_VARIABLE(0);
                auto numResults = OUTPUT_VARIABLE(1);
                auto result = OUTPUT_VARIABLE(0);
                processCondition(mode,arg,nullptr,result,numResults,scalar);
                STORE_2_RESULTS(result,numResults);
            }
*/

            return Status::OK();
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

            auto shapeScalar = ShapeBuilders::createScalarShapeInfo(block.workspace());

            return SHAPELIST(newShape, shapeScalar);
        }


    }
}

#endif