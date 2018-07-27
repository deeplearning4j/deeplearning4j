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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/roll.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void rollFunctorLinear(NDArray<T>* input, NDArray<T>* output, int shift, bool inplace){
        NDArray<T>* source = input;
        if (!inplace)
            output->assign(input);

        int fullLen = source->lengthOf();
        int actualShift = shift; // % fullLen; // shift already non-negative then
        if (actualShift < 0) {
            actualShift -= fullLen * (actualShift / fullLen - 1);
        }
        else
            actualShift %= fullLen;

        if (actualShift) {
            int shiftCount = fullLen / actualShift - 1;
            int remainShift = fullLen % actualShift; 
            
            // stage 1) swap last actualShift elements with first ones.
#pragma omp parallel for if (actualShift > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 0; e < actualShift; ++e) {
                int sourceIndex = fullLen - actualShift + e;
                nd4j::math::nd4j_swap((*output)(e), (*output)(sourceIndex));
            }

            // stage 2) swap swapped actualShift elements with rest remainShiftCount times.
#pragma omp parallel for if (shiftCount > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int count = 1; count < shiftCount; ++count) {
                for (int e = 0; e < actualShift; ++e) {
                    int destinationIndex = fullLen - (count + 1) * actualShift + e;
                    int sourceIndex = fullLen - count * actualShift + e;
                    nd4j::math::nd4j_swap((*output)(destinationIndex), (*output)(sourceIndex));
                }
            }
            
            // stage 3) swap remainer of items.
            if (remainShift && shiftCount)
            for (int i = actualShift; i < 2 * actualShift; ++i) 
                nd4j::math::nd4j_swap((*output)(i), (*output)(i + remainShift));
        }
    }

    template <typename T>
    void rollFunctorFull(NDArray<T>* input, NDArray<T>* output, int shift, std::vector<int> const& axes, bool inplace){

        if (!inplace)
            output->assign(input);

        NDArray<T>* source = input;
        for (int axe: axes) {
            if (axe == source->rankOf() - 1) {// last dimension
                std::unique_ptr<ResultSet<T>> listOfTensors(source->allTensorsAlongDimension({axe}));
                std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension({axe}));
                int fullLen = listOfTensors->size();
                int theShift = shift;
                if (theShift > 0) {
                    theShift %= fullLen;
                }
                else {
                        theShift -= fullLen * (theShift / fullLen - 1);
                }
                for (int k = 0; k < fullLen; k++) {
                    rollFunctorLinear(listOfTensors->at(k), listOfOutTensors->at(k), theShift, true);
                }
            }
            else {
                std::vector<int> dims(source->rankOf() - axe - 1);
                for (int i = 0; i < dims.size(); ++i)
                    dims[i] = axe + 1 + i;

                std::unique_ptr<ResultSet<T>> listOfTensors(source->allTensorsAlongDimension({dims}));
                std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension({dims}));
            
                int fullLen = listOfTensors->size();
                int sizeAt = input->sizeAt(axe);

                int theShift = shift;

                if (theShift > 0) {
                    theShift %= sizeAt;
                }
                else {
                    theShift -= sizeAt * (theShift / sizeAt - 1);
                }

                if (theShift) {
                    for (int dim = 0; dim < fullLen / sizeAt; ++dim) {
                        for (int e = theShift; e < sizeAt - theShift; ++e) {
                            NDArray<T>* sourceM = listOfTensors->at(dim * sizeAt + e - theShift);
                            NDArray<T>* targetM = listOfOutTensors->at(dim * sizeAt + e);
                            sourceM->swapUnsafe(*targetM);
                        }
    
                        for (int e = 0; e < theShift; ++e) {
                            int sourceIndex = dim * sizeAt + sizeAt - theShift + e;
                            NDArray<T>* sourceM = listOfTensors->at(sourceIndex);
                            NDArray<T>* targetM = listOfOutTensors->at(dim * sizeAt + e);
    
                            sourceM->swapUnsafe(*targetM);
                        }
                    }
                }
            }
            if (!inplace)
                source = output;
        }
    }

    template void rollFunctorLinear(NDArray<float>*   input, NDArray<float>*   output, int shift, bool inplace);
    template void rollFunctorLinear(NDArray<float16>* input, NDArray<float16>* output, int shift, bool inplace);
    template void rollFunctorLinear(NDArray<double>*  input, NDArray<double>*  output, int shift, bool inplace);
    template void rollFunctorFull(NDArray<float>*   input, NDArray<float>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
    template void rollFunctorFull(NDArray<float16>* input, NDArray<float16>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
    template void rollFunctorFull(NDArray<double>*  input, NDArray<double>* axisVector, int shift, std::vector<int> const& axes, bool inplace);
}
}
}