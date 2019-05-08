/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/choose.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    static nd4j::NDArray* processCondition_(int mode,nd4j::NDArray *arg, nd4j::NDArray *comp, nd4j::NDArray& compScalar);

    template <typename T>
    static T processElementCondition(int mode,T d1,T d2);


    template <typename T>
    nd4j::NDArray* processCondition_(int mode,nd4j::NDArray *arg, nd4j::NDArray *comp, nd4j::NDArray *output, nd4j::NDArray *numResult, nd4j::NDArray& compScalar) {

        //Convert to straight ndarray based on input

        int numResults = 0;
        if(comp != nullptr) {
            if (comp->isScalar()) {
                //Other input for compare could be an ndarray or a secondary scalar
                //for comparison
//                nd4j::NDArray arg1 = *arg;
//                nd4j::NDArray comp1 = *comp;
                for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                    T result2 = processElementCondition(mode, arg->e<T>(i), comp->e<T>(0));
                    if(result2 > 0) {
                        output->p(numResults, arg->e<T>(i));
                        numResults++;
                    }
                }
            } else {
                // REQUIRE_TRUE(comp.isSameShape(arg));
                //Other input for compare could be an ndarray or a secondary scalar
                //for comparison
                nd4j::NDArray arg1 = *arg;
                for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                    T result2 = processElementCondition(mode, arg->e<T>(i), compScalar.e<T>(0));
                    if(result2 > 0) {
                        output->p(numResults, arg->e<T>(i));
                        numResults++;
                    }
                }
            }

        }
        else {
    //        nd4j::NDArray arg1 = *arg;
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            for (Nd4jLong i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition(mode, arg->e<T>(i), compScalar.e<T>(0));
                if(result2 > 0) {
                    output->p(numResults, arg->e<T>(i));
                    numResults++;
                }
            }
        }

        if(numResult != nullptr)
            numResult->p(0,numResults);

        return output;

    }

    nd4j::NDArray* processCondition(int mode,nd4j::NDArray *arg, nd4j::NDArray *comp, nd4j::NDArray *output, nd4j::NDArray *numResult, nd4j::NDArray& compScalar) {
        BUILD_SINGLE_SELECTOR(arg->dataType(), return processCondition_, (mode, arg, comp, output, numResult, compScalar), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* processCondition_, (int mode,nd4j::NDArray *arg, nd4j::NDArray *comp, nd4j::NDArray *output, nd4j::NDArray *numResult, nd4j::NDArray& compScalar), FLOAT_TYPES);

    template <typename T>
    T processElementCondition(int mode,T d1,T d2) {
        T modePointer = (T ) mode;
        T input[3] = {d2, (T) EPS, (T) mode};
        T res = simdOps::MatchCondition<T,T>::op(d1, input);
        return res;

    }

    void chooseFunctorArray(NDArray* arg, NDArray* comp, int mode, NDArray* result, NDArray* numResults) {
        if(arg->isScalar() || comp->isScalar()) {
            if(arg->isScalar()) {
                processCondition(mode,comp,nullptr,result,numResults, *arg);
            }
            else {
                processCondition(mode,arg,nullptr,result,numResults, *comp);
            }
        }
        else {
            auto zero = NDArrayFactory::create<float>(0);
            processCondition(mode,arg,comp,result,numResults, zero);
        }
    }

    void chooseFunctorScalar(NDArray* arg, double scalar, int mode, NDArray* result, NDArray* numResults) {
        NDArray scalarA = NDArrayFactory::create(scalar);
        processCondition(mode, arg, nullptr,result, numResults, scalarA);
    }

}
}
}
