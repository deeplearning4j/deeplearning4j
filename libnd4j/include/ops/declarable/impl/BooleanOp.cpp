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
// Created by raver119 on 13.10.2017.
//

#include "ops/declarable/BooleanOp.h"
#include <vector>
#include <initializer_list>
#include <NDArrayFactory.h>

namespace nd4j {
    namespace ops {
        BooleanOp::BooleanOp(const char *name, int numInputs, bool scalar) : DeclarableOp::DeclarableOp(name, numInputs, scalar) {
            //
        }

        BooleanOp::~BooleanOp() {
            //
        }

        /**
        * Output shape of any BooleanOp is ALWAYS scalar
        */
        ShapeList *BooleanOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            Nd4jLong *shapeNew;
            ALLOCATE(shapeNew, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
            shapeNew[0] = 2;
            shapeNew[1] = 1;
            shapeNew[2] = 1;
            shapeNew[3] = 1;
            shapeNew[4] = 1;
            shapeNew[5] = 0;
            shapeNew[6] = 1;
            shapeNew[7] = 99;

            return SHAPELIST(shapeNew);
        }

        bool BooleanOp::evaluate(nd4j::graph::Context &block) {
            // check if scalar or not

            // validation?

            Nd4jStatus status = this->validateNonEmptyInput(block);
            if (status != ND4J_STATUS_OK) {
                nd4j_printf("Inputs should be not empty for BooleanOps","");
                throw std::runtime_error("Bad inputs");
            }

            status = this->validateAndExecute(block);
            if (status == ND4J_STATUS_TRUE)
                return true;
            else if (status == ND4J_STATUS_FALSE)
                return false;
            else {
                nd4j_printf("Got error %i during [%s] evaluation: ", (int) status, this->getOpDescriptor()->getOpName()->c_str());
                throw std::runtime_error("Internal error");
            }
        }

        bool BooleanOp::evaluate(std::initializer_list<nd4j::NDArray *> args) {
            std::vector<nd4j::NDArray *> vec(args);
            return this->evaluate(vec);
        }

        bool BooleanOp::prepareOutputs(Context& ctx) {

            auto variableSpace = ctx.getVariableSpace();

            for (int e = 0; e < this->getOpDescriptor()->getNumberOfOutputs(); e++) {
                std::pair<int, int> pair(ctx.nodeId(), e);

                if (!variableSpace->hasVariable(pair))
                    variableSpace->putVariable(pair, new Variable());

                auto var = ctx.variable(pair);

                if (var->getNDArray() == nullptr) {
                    var->setNDArray(NDArrayFactory::create('c', {1, 1}, DataType_BOOL, ctx.getWorkspace()));
                    var->markRemovable(true);
                }
            }

            return true;
        }

        Nd4jStatus nd4j::ops::BooleanOp::execute(Context* block)  {

            // basic validation: ensure inputs are set
            REQUIRE_OK(this->validateNonEmptyInput(*block));

            // ensure number of IArgs, TArgs match our expectations
            REQUIRE_OK(this->validateArguments(*block));

            // this method will allocate output NDArrays for this op
            this->prepareOutputs(*block);

            auto timeStart = std::chrono::system_clock::now();

            Nd4jStatus status = this->validateAndExecute(*block);

            auto timeEnd = std::chrono::system_clock::now();
            auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
            block->setInnerTime(outerTime);

            // basically we're should be putting 0.0 as FALSE, and any non-0.0 value will be treated as TRUE
            std::pair<int,int> p(block->nodeId(), 0);
            auto var = block->variable(p);
            var->getNDArray()->p(Nd4jLong(0), status == ND4J_STATUS_TRUE ?  1.0f : 0.0f);

            if (status == ND4J_STATUS_FALSE || status == ND4J_STATUS_TRUE)
                return ND4J_STATUS_OK;
            
            nd4j_printf("%s: node_%i got unexpected result instead of boolean: [%i]\n", this->getOpName()->c_str(), block->nodeId(), status);
            return ND4J_STATUS_KERNEL_FAILURE;
        }

        bool BooleanOp::evaluate(std::vector<nd4j::NDArray *> &args) {
            VariableSpace variableSpace;

            int cnt = -1;
            std::vector<int> in;
            for (auto v: args) {
                auto var = new Variable(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            Context block(1, &variableSpace, false);
            block.fillInputs(in);

            return this->evaluate(block);
        }
    }
}

