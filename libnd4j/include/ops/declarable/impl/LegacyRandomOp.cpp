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
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyRandomOp.h>
#include <helpers/RandomLauncher.h>
#include <NativeOpExcutioner.h>
#include <NDArrayFactory.h>
#include <Status.h>


namespace nd4j {
    namespace ops {
        LegacyRandomOp::LegacyRandomOp() : LegacyOp::LegacyOp(1) {
            // just a no-op
        }

        LegacyRandomOp::LegacyRandomOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
            // just a no-op
        }

        LegacyOp* LegacyRandomOp::clone() {
            return new LegacyRandomOp(this->_opNum);
        }

        template <typename T>
        Nd4jStatus LegacyRandomOp::validateAndExecute_(Context &block) {
            auto input = INPUT_VARIABLE(0);

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            /*
                (0, randomOps::UniformDistribution) ,\
                (1, randomOps::DropOut) ,\
                (2, randomOps::DropOutInverted) ,\
                (3, randomOps::ProbablisticMerge) ,\
                (4, randomOps::Linspace) ,\
                (5, randomOps::Choice) ,\
                (6, randomOps::GaussianDistribution) ,\
                (7, randomOps::BernoulliDistribution) ,\
                (8, randomOps::BinomialDistribution),\
                (9, randomOps::BinomialDistributionEx),\
                (10, randomOps::LogNormalDistribution) ,\
                (11, randomOps::TruncatedNormalDistribution) ,\
                (12, randomOps::AlphaDropOut)
            */
            switch(opNum) {
                case 0: {
                    // uniform distribution
                    T from, to;
                    if (block.width() > 2) {
                        auto arg1 = INPUT_VARIABLE(1);
                        auto arg2 = INPUT_VARIABLE(2);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "Uniform: Second argument must be scalar");
                        REQUIRE_TRUE(arg2->isScalar(), 0, "Uniform: Third argument must be scalar");

                        from = arg1->getScalar<T>(0);
                        to = arg2->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 2) {
                        from = T_ARG(0);
                        to = T_ARG(1);
                    } else {
                        REQUIRE_TRUE(false, 0, "Uniform requires either TArgs or 3 arguments to be present");
                    }

                    REQUIRE_TRUE(input->isVector(), 0, "Uniform requires pure shape as first argument");
                    std::vector<Nd4jLong> shape(input->lengthOf());
                    for (int e = 0; e < input->lengthOf(); e++)
                        shape[e] = input->getScalar<Nd4jLong>(e);

                    auto z = new NDArray('c', shape, block.dataType(), block.getWorkspace());

                    RandomLauncher::fillUniform(block.getRNG(), z, from, to);

                    // FIXME:
                    OVERWRITE_RESULT(z);
                }
                    break;
                case 1: {
                    auto z = OUTPUT_VARIABLE(0);

                    T prob;
                    if (block.width() > 1) {
                        auto arg = INPUT_VARIABLE(1);
                        REQUIRE_TRUE(arg->isScalar(), 0, "DropOut: Second argument must be scalar");

                        prob = arg->getScalar<T>(0);
                    } else if (block.getTArguments()->size() > 0) {
                        prob = T_ARG(0);
                    } else {
                        REQUIRE_TRUE(false, 0, "DropOut requires either TArgs or second argument to be present");
                    }

                    if (!block.isInplace())
                        z->assign(input);

                    RandomLauncher::applyDropOut(block.getRNG(), z, prob);
                }
                    break;
                case 2: {
                    auto z = OUTPUT_VARIABLE(0);

                    T prob;
                    if (block.width() > 1) {
                        auto arg = INPUT_VARIABLE(1);
                        REQUIRE_TRUE(arg->isScalar(), 0, "InvertedDropOut: Second argument must be scalar");

                        prob = arg->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 1) {
                        prob = T_ARG(0);
                    } else {
                        REQUIRE_TRUE(false, 0, "InvertedDropOut requires either TArgs or second argument to be present");
                    }

                    if (!block.isInplace())
                        z->assign(input);

                    RandomLauncher::applyInvertedDropOut(block.getRNG(), z, prob);
                }
                    break;
                case 6: {
                    // gaussian distribution
                    T mean, stdev;
                    if (block.width() > 2) {
                        auto arg1 = INPUT_VARIABLE(1);
                        auto arg2 = INPUT_VARIABLE(2);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "Gaussian: Second argument must be scalar");
                        REQUIRE_TRUE(arg2->isScalar(), 0, "Gaussian: Third argument must be scalar");

                        mean = arg1->getScalar<T>(0);
                        stdev = arg2->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 2) {
                        mean = T_ARG(0);
                        stdev = T_ARG(1);
                    } else {
                        REQUIRE_TRUE(false, 0, "Gaussian requires either TArgs or 3 arguments to be present");
                    }

                    REQUIRE_TRUE(input->isVector(), 0, "Gaussian requires pure shape as first argument");

                    std::vector<Nd4jLong> shape(input->lengthOf());
                    for (int e = 0; e < input->lengthOf(); e++)
                        shape[e] = input->getScalar<Nd4jLong>(e);

                    auto z = NDArrayFactory::create<T>('c', shape, block.getWorkspace());

                    RandomLauncher::fillGaussian(block.getRNG(), z, mean, stdev);

                    // FIXME: !!
                    OVERWRITE_RESULT(z);
                }
                    break;
                case 7: {
                    // bernoulli distribution
                    T prob;
                    if (block.width() > 1) {
                        auto arg1 = INPUT_VARIABLE(1);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "Bernoulli: Second argument must be scalar");

                        prob = arg1->getScalar<T>(0);
                    } else if (block.getTArguments()->size() > 0) {
                        prob = T_ARG(0);
                    } else {
                        REQUIRE_TRUE(false, 0, "Bernoulli requires either 1 TArg or 2 arguments to be present");
                    }

                    REQUIRE_TRUE(input->isVector(), 0, "Bernoulli requires pure shape as first argument");

                    std::vector<Nd4jLong> shape(input->lengthOf());
                    for (int e = 0; e < input->lengthOf(); e++)
                        shape[e] = input->getScalar<Nd4jLong>(e);

                    auto z = NDArrayFactory::create<T>('c', shape, block.getWorkspace());

                    RandomLauncher::fillBernoulli(block.getRNG(), z, prob);

                    // FIXME:
                    OVERWRITE_RESULT(z);
                }
                    break;
                case 9: {
                    // BinomialEx distribution
                    T prob;
                    int trials;
                    if (block.width() > 2) {
                        auto arg1 = INPUT_VARIABLE(1);
                        auto arg2 = INPUT_VARIABLE(2);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "Binomial: Second argument must be scalar");
                        REQUIRE_TRUE(arg2->isScalar(), 0, "Binomial: Third argument must be scalar");

                        trials = arg1->getScalar<int>(0);
                        prob = arg2->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 1 && block.getIArguments()->size() == 1) {
                        trials = INT_ARG(0);
                        prob = T_ARG(0);
                    } else {
                        REQUIRE_TRUE(false, 0, "Binomial requires either TArgs/IArgs or 3 arguments to be present");
                    }

                    REQUIRE_TRUE(input->isVector(), 0, "Binomial requires pure shape as first argument");

                    std::vector<Nd4jLong> shape(input->lengthOf());
                    for (int e = 0; e < input->lengthOf(); e++)
                        shape[e] = input->getScalar<Nd4jLong>(e);

                    auto z = new NDArray('c', shape, block.dataType(), block.getWorkspace());

                    RandomLauncher::fillBinomial(block.getRNG(), z, trials, prob);

                    // FIXME: !!!
                    OVERWRITE_RESULT(z);
                }
                    break;
                case 10: {
                    // lognorm distribution
                    T mean, stdev;
                    if (block.width() > 2) {
                        auto arg1 = INPUT_VARIABLE(1);
                        auto arg2 = INPUT_VARIABLE(2);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "LogNormal: Second argument must be scalar");
                        REQUIRE_TRUE(arg2->isScalar(), 0, "LogNormal: Third argument must be scalar");

                        mean = arg1->getScalar<T>(0);
                        stdev = arg2->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 2) {
                        mean = T_ARG(0);
                        stdev = T_ARG(1);
                    } else {
                        REQUIRE_TRUE(false, 0, "LogNormal requires either TArgs or 3 arguments to be present");
                    }

                    REQUIRE_TRUE(input->isVector(), 0, "LogNormal requires pure shape as first argument");

                    std::vector<Nd4jLong> shape(input->lengthOf());
                    for (int e = 0; e < input->lengthOf(); e++)
                        shape[e] = input->getScalar<Nd4jLong>(e);

                    auto z = NDArrayFactory::create<T>('c', shape, block.getWorkspace());

                    RandomLauncher::fillLogNormal(block.getRNG(), z, mean, stdev);

                    // FIXME: !!
                    OVERWRITE_RESULT(z);
                }
                    break;
                case 11: {
                    // truncated norm distribution
                    T mean, stdev;
                    if (block.width() > 2) {
                        auto arg1 = INPUT_VARIABLE(1);
                        auto arg2 = INPUT_VARIABLE(2);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "TruncatedNormal: Second argument must be scalar");
                        REQUIRE_TRUE(arg2->isScalar(), 0, "TruncatedNormal: Third argument must be scalar");

                        mean = arg1->getScalar<T>(0);
                        stdev = arg2->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 2) {
                        mean = T_ARG(0);
                        stdev = T_ARG(1);
                    } else {
                        REQUIRE_TRUE(false, 0, "TruncatedNormal requires either TArgs or 3 arguments to be present");
                    }

                    REQUIRE_TRUE(input->isVector(), 0, "TruncatedNormal requires pure shape as first argument");

                    std::vector<Nd4jLong> shape(input->lengthOf());
                    for (int e = 0; e < input->lengthOf(); e++)
                        shape[e] = input->getScalar<Nd4jLong>(e);

                    auto z = NDArrayFactory::create<T>('c', shape, block.getWorkspace());

                    RandomLauncher::fillTruncatedNormal(block.getRNG(), z, mean, stdev);

                    // FIXME: !!!
                    OVERWRITE_RESULT(z);
                }
                    break;
                case 12: {
                    auto z = OUTPUT_VARIABLE(0);

                    T prob, a, b, pa;
                    if (block.width() > 4) {
                        auto arg1 = INPUT_VARIABLE(1);
                        auto arg2 = INPUT_VARIABLE(2);
                        auto arg3 = INPUT_VARIABLE(3);
                        auto arg4 = INPUT_VARIABLE(4);
                        REQUIRE_TRUE(arg1->isScalar(), 0, "AlphaDropOut: Second argument must be scalar");
                        REQUIRE_TRUE(arg2->isScalar(), 0, "AlphaDropOut: Third argument must be scalar");
                        REQUIRE_TRUE(arg3->isScalar(), 0, "AlphaDropOut: Fourth argument must be scalar");
                        REQUIRE_TRUE(arg4->isScalar(), 0, "AlphaDropOut: Fifth argument must be scalar");

                        prob = arg1->getScalar<T>(0);
                        a = arg2->getScalar<T>(0);
                        b = arg3->getScalar<T>(0);
                        pa = arg4->getScalar<T>(0);
                    } else if (block.getTArguments()->size() == 4) {
                        prob = T_ARG(0);
                        a = T_ARG(1);
                        b = T_ARG(2);
                        pa = T_ARG(3);
                    } else {
                        REQUIRE_TRUE(false, 0, "AlphaDropOut requires either TArgs or 5 arguments to be present");
                    }

                    if (!block.isInplace())
                        z->assign(input);

                    RandomLauncher::applyAlphaDropOut(block.getRNG(), z, prob, a, b, pa);
                }
                    break;
                default: {
                    nd4j_printf("Unknown random op requested: [%i]\n", opNum);
                    return ND4J_STATUS_KERNEL_FAILURE;
                }
            }

            return Status::OK();
        }

        Nd4jStatus LegacyRandomOp::validateAndExecute(Context &block) {
            REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RNG should be provided for LegacyRandomOp, but got NULL instead at node_%i", block.nodeId())

            auto z = OUTPUT_VARIABLE(0);
            BUILD_SINGLE_SELECTOR(z->dataType(), return validateAndExecute_, (block), FLOAT_TYPES);
        }

        /**
        * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and col2im. 
        * But these ops already have CustomOp implementations.
        *
        */
        ShapeList *LegacyRandomOp::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }

        Nd4jStatus LegacyRandomOp::execute(Context* block) {
            return DeclarableOp::execute(block);
        }

        nd4j::ResultSet*  LegacyRandomOp::execute(nd4j::random::RandomBuffer* rng, std::initializer_list<NDArray*> inputs, std::initializer_list<double> tArgs, std::initializer_list<int> iArgs, bool isInplace) {
            std::vector<NDArray*> ins(inputs);
            std::vector<double> tas(tArgs);
            std::vector<int> ias(iArgs);
            return this->execute(rng, ins, tas, ias, isInplace);
        }

        nd4j::ResultSet*  LegacyRandomOp::execute(nd4j::random::RandomBuffer* rng, std::vector<NDArray*>& inputs, std::vector<double>& tArgs, std::vector<int>& iArgs, bool isInplace) {
            VariableSpace variableSpace;
            auto arrayList = new ResultSet();
            //ResultSet arrayList;

            if (isInplace)
                arrayList->setNonRemovable();

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
                if (v == nullptr)
                    continue;

                auto var = new Variable(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            Context block(1, &variableSpace, false);
            block.setRNG(rng);
            block.fillInputs(in);
            block.markInplace(isInplace);

            for (int e = 0; e < tArgs.size(); e++)
                block.getTArguments()->emplace_back(tArgs.at(e));


            for (int e = 0; e < iArgs.size(); e++)
                block.getIArguments()->emplace_back(iArgs.at(e));

            Nd4jStatus status = this->execute(&block);
            arrayList->setStatus(status);
            if (status != ND4J_STATUS_OK)
                return arrayList;


            for (int e = 0; e < 65536; e++) {
                std::pair<int,int> pair(1, e);
                if (variableSpace.hasVariable(pair)) {
                    auto var = variableSpace.getVariable(pair);
                    auto arr = var->getNDArray();
                    if (!arr->isAttached()) {
                        var->markRemovable(false);
                        arrayList->push_back(arr);
                    } else {
                        arrayList->push_back(arr->detach());
                    }
                } else
                    break;
            }

            return arrayList;
        }

        BUILD_SINGLE_TEMPLATE(template Nd4jStatus LegacyRandomOp::validateAndExecute_, (Context&), FLOAT_TYPES);
    }
}