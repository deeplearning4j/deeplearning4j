/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/NDArrayFactory.h>
#include <helpers/RandomLauncher.h>
#include <legacy/NativeOpExecutioner.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/OpRegistrator.h>

namespace sd {
namespace ops {
LegacyRandomOp::LegacyRandomOp() : LegacyOp(1) {
  // just a no-op
}

LegacyRandomOp::LegacyRandomOp(int opNum) : LegacyOp(1, opNum) {
  // just a no-op
}

LegacyOp* LegacyRandomOp::clone() { return new LegacyRandomOp(this->_opNum); }

template <typename T>
Status LegacyRandomOp::validateAndExecute_(Context& block) {
  auto input = INPUT_VARIABLE(0);

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  switch (opNum) {
    case random::UniformDistribution: {
      // uniform distribution
      T from, to;
      if (block.width() > 2) {
        auto arg1 = INPUT_VARIABLE(1);
        auto arg2 = INPUT_VARIABLE(2);
        REQUIRE_TRUE(arg1->isScalar(), 0, "Uniform: Second argument must be scalar");
        REQUIRE_TRUE(arg2->isScalar(), 0, "Uniform: Third argument must be scalar");

        from = arg1->e<T>(0);
        to = arg2->e<T>(0);
      } else if (block.getTArguments()->size() == 2) {
        from = T_ARG(0);
        to = T_ARG(1);
      } else {
        REQUIRE_TRUE(false, 0, "Uniform requires either TArgs or 3 arguments to be present");
      }

      auto z = OUTPUT_VARIABLE(0);  // NDArrayFactory::create_<T>('c', shape, block.getWorkspace());

      RandomLauncher::fillUniform(block.launchContext(), block.randomGenerator(), z, from, to);

      // FIXME:
      // OVERWRITE_RESULT(z);
    } break;
    case random::DropOut: {
      auto z = OUTPUT_VARIABLE(0);

      T prob;
      if (block.width() > 1) {
        auto arg = INPUT_VARIABLE(1);
        REQUIRE_TRUE(arg->isScalar(), 0, "DropOut: Second argument must be scalar");

        prob = arg->e<T>(0);
      } else if (block.getTArguments()->size() > 0) {
        prob = T_ARG(0);
      } else {
        REQUIRE_TRUE(false, 0, "DropOut requires either TArgs or second argument to be present");
      }

      if (!block.isInplace()) z->assign(*input);

      RandomLauncher::applyDropOut(block.launchContext(), block.randomGenerator(), z, prob);
    } break;
#if NOT_EXCLUDED(OP_dropout)
    case random::DropOutInverted: {
      auto z = OUTPUT_VARIABLE(0);
      dropout op;
      return op.execute(&block);
    } break;
#endif
    case random::GaussianDistribution: {
      // gaussian distribution
      T mean, stdev;
      if (block.width() > 2) {
        auto arg1 = INPUT_VARIABLE(1);
        auto arg2 = INPUT_VARIABLE(2);
        REQUIRE_TRUE(arg1->isScalar(), 0, "Gaussian: Second argument must be scalar");
        REQUIRE_TRUE(arg2->isScalar(), 0, "Gaussian: Third argument must be scalar");

        mean = arg1->e<T>(0);
        stdev = arg2->e<T>(0);
      } else if (block.getTArguments()->size() == 2) {
        mean = T_ARG(0);
        stdev = T_ARG(1);
      } else {
        REQUIRE_TRUE(false, 0, "Gaussian requires either TArgs or 3 arguments to be present");
      }

      REQUIRE_TRUE(input->isVector(), 0, "Gaussian requires pure shape as first argument");

      std::vector<LongType> shape(input->lengthOf());
      for (int e = 0; e < input->lengthOf(); e++) shape[e] = input->e<LongType>(e);

      auto z = OUTPUT_VARIABLE(0);

      RandomLauncher::fillGaussian(block.launchContext(), block.randomGenerator(), z, mean, stdev);


    } break;
    case random::BernoulliDistribution: {
      // bernoulli distribution
      T prob;
      if (block.width() > 1) {
        auto arg1 = INPUT_VARIABLE(1);
        REQUIRE_TRUE(arg1->isScalar(), 0, "Bernoulli: Second argument must be scalar");

        prob = arg1->e<T>(0);
      } else if (block.getTArguments()->size() > 0) {
        prob = T_ARG(0);
      } else {
        REQUIRE_TRUE(false, 0, "Bernoulli requires either 1 TArg or 2 arguments to be present");
      }

      REQUIRE_TRUE(input->isVector(), 0, "Bernoulli requires pure shape as first argument");

      std::vector<LongType> shape(input->lengthOf());
      for (int e = 0; e < input->lengthOf(); e++) shape[e] = input->e<LongType>(e);

      auto z = OUTPUT_VARIABLE(0);  // NDArrayFactory::create_<T>('c', shape, block.getWorkspace());

      RandomLauncher::fillBernoulli(block.launchContext(), block.randomGenerator(), z, prob);


    } break;
    case random::BinomialDistributionEx: {
      // BinomialEx distribution
      T prob;
      int trials;
      if (block.width() > 2) {
        auto arg1 = INPUT_VARIABLE(1);
        auto arg2 = INPUT_VARIABLE(2);
        REQUIRE_TRUE(arg1->isScalar(), 0, "Binomial: Second argument must be scalar");
        REQUIRE_TRUE(arg2->isScalar(), 0, "Binomial: Third argument must be scalar");

        trials = arg1->e<int>(0);
        prob = arg2->e<T>(0);
      } else if (block.getTArguments()->size() == 1 && block.getIArguments()->size() == 1) {
        trials = INT_ARG(0);
        prob = T_ARG(0);
      } else {
        REQUIRE_TRUE(false, 0, "Binomial requires either TArgs/IArgs or 3 arguments to be present");
      }

      REQUIRE_TRUE(input->isVector(), 0, "Binomial requires pure shape as first argument");

      std::vector<LongType> shape(input->lengthOf());
      for (int e = 0; e < input->lengthOf(); e++) shape[e] = input->e<LongType>(e);

      auto z = OUTPUT_VARIABLE(0);
      RandomLauncher::fillBinomial(block.launchContext(), block.randomGenerator(), z, trials, prob);

    } break;
    case random::LogNormalDistribution: {
      // lognorm distribution
      T mean, stdev;
      if (block.width() > 2) {
        auto arg1 = INPUT_VARIABLE(1);
        auto arg2 = INPUT_VARIABLE(2);
        REQUIRE_TRUE(arg1->isScalar(), 0, "LogNormal: Second argument must be scalar");
        REQUIRE_TRUE(arg2->isScalar(), 0, "LogNormal: Third argument must be scalar");

        mean = arg1->e<T>(0);
        stdev = arg2->e<T>(0);
      } else if (block.getTArguments()->size() == 2) {
        mean = T_ARG(0);
        stdev = T_ARG(1);
      } else {
        REQUIRE_TRUE(false, 0, "LogNormal requires either TArgs or 3 arguments to be present");
      }

      REQUIRE_TRUE(input->isVector(), 0, "LogNormal requires pure shape as first argument");

      std::vector<LongType> shape(input->lengthOf());
      for (int e = 0; e < input->lengthOf(); e++) shape[e] = input->e<LongType>(e);

      auto z = OUTPUT_VARIABLE(0);
      RandomLauncher::fillLogNormal(block.launchContext(), block.randomGenerator(), z, mean, stdev);

    } break;
    case random::TruncatedNormalDistribution: {
      // truncated norm distribution
      T mean, stdev;
      if (block.width() > 2) {
        auto arg1 = INPUT_VARIABLE(1);
        auto arg2 = INPUT_VARIABLE(2);
        REQUIRE_TRUE(arg1->isScalar(), 0, "TruncatedNormal: Second argument must be scalar");
        REQUIRE_TRUE(arg2->isScalar(), 0, "TruncatedNormal: Third argument must be scalar");

        mean = arg1->e<T>(0);
        stdev = arg2->e<T>(0);
      } else if (block.getTArguments()->size() == 2) {
        mean = T_ARG(0);
        stdev = T_ARG(1);
      } else {
        REQUIRE_TRUE(false, 0, "TruncatedNormal requires either TArgs or 3 arguments to be present");
      }

      REQUIRE_TRUE(input->isVector(), 0, "TruncatedNormal requires pure shape as first argument");

      std::vector<LongType> shape(input->lengthOf());
      for (int e = 0; e < input->lengthOf(); e++) shape[e] = input->e<LongType>(e);

      auto z = OUTPUT_VARIABLE(0);

      RandomLauncher::fillTruncatedNormal(block.launchContext(), block.randomGenerator(), z, mean, stdev);
    } break;
    case random::AlphaDropOut: {
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

        prob = arg1->e<T>(0);
        a = arg2->e<T>(0);
        b = arg3->e<T>(0);
        pa = arg4->e<T>(0);
      } else if (block.getTArguments()->size() == 4) {
        prob = T_ARG(0);
        a = T_ARG(1);
        b = T_ARG(2);
        pa = T_ARG(3);
      } else {
        REQUIRE_TRUE(false, 0, "AlphaDropOut requires either TArgs or 5 arguments to be present");
      }

      if (!block.isInplace()) z->assign(*input);

      RandomLauncher::applyAlphaDropOut(block.launchContext(), block.randomGenerator(), z, prob, a, b, pa);
    } break;
    case random::Linspace: {
      auto z = OUTPUT_VARIABLE(0);
      auto start = INPUT_VARIABLE(0);
      auto finish = INPUT_VARIABLE(1);
      auto numOfElements = INPUT_VARIABLE(2);

      z->linspace(start->e<double>(0),
                  (finish->e<double>(0) - start->e<double>(0)) / (numOfElements->e<LongType>(0) - 1.));
    } break;
    default: {
      sd_printf("Unknown random op requested: [%i]\n", opNum);
      return Status::KERNEL_FAILURE;
    }
  }

  traceExecIfNeeded(block);

  return Status::OK;
}

Status LegacyRandomOp::validateAndExecute(Context& block) {
  auto z = OUTPUT_VARIABLE(0);
  BUILD_SINGLE_SELECTOR(z->dataType(), return validateAndExecute_, (block), SD_FLOAT_TYPES);
}

/**
 * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and
 * col2im. But these ops already have CustomOp implementations.
 *
 */
ShapeList* LegacyRandomOp::calculateOutputShape(ShapeList* inputShape, Context& block) {
  auto inShape = inputShape->at(0);
  auto xType = ArrayOptions::dataType(inShape);
  LongType* newShape;
  if (DataTypeUtils::isR(xType)) {
    COPY_SHAPE(inShape, newShape);

    return SHAPELIST(CONSTANT(newShape));
  } else if (DataTypeUtils::isZ(xType)) {
    auto zShapeArr = INPUT_VARIABLE(0);
    auto zShapeVector = zShapeArr->asVectorT<LongType>();
    auto dtype = block.dataType();

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', zShapeVector));
  } else
    THROW_EXCEPTION("LegacyRandomOp: Unknown input data type!");
}

Status LegacyRandomOp::execute(Context* block) { return DeclarableOp::execute(block); }

ResultSet LegacyRandomOp::execute(RandomGenerator& rng, std::initializer_list<NDArray*> inputs,
                                  std::initializer_list<double> tArgs, std::initializer_list<int> iArgs,
                                  bool isInplace) {
  std::vector<NDArray*> ins(inputs);
  std::vector<double> tas(tArgs);
  std::vector<int> ias(iArgs);
  return this->execute(rng, ins, tas, ias, isInplace);
}

ResultSet LegacyRandomOp::execute(RandomGenerator& rng, std::vector<NDArray*>& inputs, std::vector<double>& tArgs,
                                  std::vector<int>& iArgs, bool isInplace) {
  VariableSpace variableSpace;
  ResultSet arrayList;

  if (isInplace) arrayList.setNonRemovable();

  int cnt = -1;
  std::vector<int> in;
  for (auto v : inputs) {
    if (v == nullptr) continue;

    auto var = new Variable(v);
    var->markRemovable(false);
    in.push_back(cnt);
    variableSpace.putVariable(cnt--, var);
  }

  Context block(1, &variableSpace, false);
  // FIX ME: implement setRng method
  block.setRng(rng);
  block.fillInputs(in);
  block.markInplace(isInplace);

  for (int e = 0; e < tArgs.size(); e++) block.getTArguments()->emplace_back(tArgs.at(e));

  for (int e = 0; e < iArgs.size(); e++) block.getIArguments()->emplace_back(iArgs.at(e));

  Status status = this->execute(&block);
  arrayList.setStatus(status);
  if (status != Status::OK) return arrayList;

  for (int e = 0; e < DataTypeUtils::max<int>(); e++) {
    std::pair<int, int> pair(1, e);
    if (variableSpace.hasVariable(pair)) {
      auto var = variableSpace.getVariable(pair);
      auto arr = var->getNDArray();
      if (!arr->isAttached()) {
        var->markRemovable(false);
        arrayList.push_back(arr);
      } else {
        arrayList.push_back(arr->detach());
      }
    } else
      break;
  }

  return arrayList;
}

Status LegacyRandomOp::validateDataTypes(Context& block) {
  if (block.isFastPath()) {
    // in this case we'll roll through pre-defined outputs
    auto fpo = block.fastpath_out();
    for (auto v : fpo) {
      if (v != nullptr) {
        if (!v->isR()) return Status::BAD_ARGUMENTS;
      }
    }
  } else {
    std::pair<int, int> pair(block.nodeId(), 0);
    if (block.getVariableSpace()->hasVariable(pair)) {
      auto var = block.variable(pair);
      if (!var->hasNDArray()) return Status::BAD_ARGUMENTS;

      auto arr = var->getNDArray();
      if (!arr->isR()) return Status::BAD_ARGUMENTS;
    }
  }

  return Status::OK;
}

BUILD_SINGLE_TEMPLATE(template sd::Status LegacyRandomOp::validateAndExecute_, (Context&), SD_FLOAT_TYPES);
}  // namespace ops
}  // namespace sd
