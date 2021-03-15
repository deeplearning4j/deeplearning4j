/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_multinomial)

#include <ops/declarable/CustomOperations.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/helpers/random.h>

namespace sd {
    namespace ops {
        ///////////////////////
        /**
         * multinomial (categorical) random generator
         * takes 2D ndarray with logits with shape [batch_size (N), num_classes (K)]
         * and array with one scalar value of samples number, number of independent samples to draw for each experiment 1,N.
         * represents the unnormalized log-probabilities for all classes.
         * Int arguments: 0 - optional argument, corresponds to dimension with batch_size
         * Int arguments: 1 - optional argument, integer type to use for the output. Default int64.
         */
         // used https://en.wikipedia.org/wiki/Categorical_distribution
         // methods: gumbel trick + softmax + argmax
        CUSTOM_OP_IMPL(random_multinomial, 2, 1, false, 0, 0) {
            
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_NULLIFIED(0);
            auto inputSamples = INPUT_VARIABLE(1);


            REQUIRE_TRUE(!input->isEmpty(), 0, "RANDOM_MULTINOMIAL OP: Have to be provided at least one logits. ");

            REQUIRE_TRUE(inputSamples->lengthOf() == 1, 0, "RANDOM_MULTINOMIAL OP: Have to be specified at least one sample,"
                " but got no argumets instead.");
            
            Nd4jLong numOfSamples = static_cast<Nd4jLong>(inputSamples->e<int>(0));
            // do nothing if number of samples = 0
            if (0 == numOfSamples)
                return Status::OK();

            REQUIRE_TRUE(numOfSamples > 0, 0, "RANDOM_MULTINOMIAL OP: Number of samples should be greater then 0, got %i. ", numOfSamples);

            const int rank = input->rankOf();
            REQUIRE_TRUE(rank == 2, 0, "RANDOM_MULTINOMIAL OP: Logits should be a matrix with rank = 2, but got instead rank = %i.", rank);

            const int argSize = block.getIArguments()->size();
            const int dimC = argSize > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;

            auto dimA = (0 == dimC) ? 1 : 0;
            if (1 == input->sizeAt(dimA)) {
                *output = 0;
                return Status::OK();
            }

            auto rng = block.randomGenerator();
            helpers::fillRandomMultiNomial(block.launchContext(), rng, *input, *output, numOfSamples, dimC);
            return Status::OK();
        }


        DECLARE_SHAPE_FN(random_multinomial) {

            auto input = INPUT_VARIABLE(0);
            auto inputSamples = INPUT_VARIABLE(1);

            REQUIRE_TRUE(inputSamples->lengthOf() == 1, 0, "RANDOM_MULTINOMIAL OP: Have to be specified at least one sample,"
                " but got no argumets instead.");

            Nd4jLong numOfSamples = static_cast<Nd4jLong>(inputSamples->e<int>(0));

            REQUIRE_TRUE(numOfSamples > 0, 0, "RANDOM_MULTINOMIAL OP: Number of samples should be greater then 0, got %i. ", numOfSamples);

            const int rank = input->rankOf();
            REQUIRE_TRUE(rank == 2, 0, "RANDOM_MULTINOMIAL OP: Logits should be a matrix with rank = 2, but got instead rank = %i.", rank);

            const int argSize = block.getIArguments()->size();
            const int dimC = argSize > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;
           
            auto nShape = input->getShapeAsVector();
            auto dimA = (0 == dimC) ? 1 : 0;
            nShape[dimA] = numOfSamples;

            DataType nType = (argSize > 1) ? ( INT_ARG(1) >= 0 ? static_cast<DataType>(INT_ARG(1)) : sd::DataType::INT64) : sd::DataType::INT64;
            return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(nType, input->ordering(), nShape));
        }
        
        DECLARE_TYPES(random_multinomial) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, { ALL_FLOATS, ALL_INTS })
                    ->setAllowedInputTypes(1, { sd::DataType::INT32 })
                    ->setAllowedOutputTypes(0, { ALL_INDICES });
        }
    }
}

#endif