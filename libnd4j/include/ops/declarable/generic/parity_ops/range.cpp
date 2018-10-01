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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_range)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/range.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(range, -2, 1, false, -2, -2) {
    auto output = OUTPUT_VARIABLE(0);

    const int numInArrs = block.width();
    const int numTArgs  = block.getTArguments()->size();
    const int numIArgs  = block.getIArguments()->size();

    NDArray *s = nullptr;
    NDArray *d = nullptr;

    bool localS = false;
    bool localD = false;
    // FIXME: this op should be fully moved to helpers


    if (numIArgs > 0) {
    
        if(numIArgs == 1) {
          //  limit = INT_ARG(0);
        } else if(numIArgs == 2) {
            s = NDArrayFactory::create_(INT_ARG(0), block.workspace());
            //limit = INT_ARG(1);
            d = NDArrayFactory::create_(1, block.workspace());
        }
        else {
            s = NDArrayFactory::create_(INT_ARG(0), block.workspace());
            //limit = INT_ARG(1);
            d = NDArrayFactory::create_(INT_ARG(2), block.workspace());
        }

        localS = true;
        localD = true;
    }         
    else if (numTArgs > 0) {

        if(numTArgs == 1) {
            //limit = T_ARG(0);
            s = NDArrayFactory::create_(0.0f, block.workspace());
            d = NDArrayFactory::create_(1.0f, block.workspace());
        } else if(numTArgs == 2) {
            s = NDArrayFactory::create_(T_ARG(0), block.workspace());
            //limit = T_ARG(1);
            d = NDArrayFactory::create_(1.0f, block.workspace());
        }
        else {
            s = NDArrayFactory::create_(T_ARG(0), block.workspace());
            //limit = T_ARG(1);
            d = NDArrayFactory::create_(T_ARG(2), block.workspace());
        }

        localS = true;
        localD = true;
    } 
    else if (numInArrs > 0) {

        if(numInArrs == 1) {
            //limit = (*INPUT_VARIABLE(0))(0.);
            if (output->isR()) {
                s = NDArrayFactory::create_(0.0f, block.workspace());
                d = NDArrayFactory::create_(1.0f, block.workspace());
            } else {
                s = NDArrayFactory::create_(0, block.workspace());
                d = NDArrayFactory::create_(1, block.workspace());
            }
            localS = true;
            localD = true;
        } else if(numInArrs == 2) {
            s = INPUT_VARIABLE(0);
            //limit = (*INPUT_VARIABLE(1))(0.);
            if (output->isR()) {
                d = NDArrayFactory::create_(1.0f, block.workspace());
            } else {
                d = NDArrayFactory::create_(1, block.workspace());
            }
            localD = true;
        }
        else {
            s = INPUT_VARIABLE(0);
            //limit = (*INPUT_VARIABLE(1))(0.);
            d = INPUT_VARIABLE(2);
        }
    } 
    else
        REQUIRE_TRUE(false, 0, "CUSTOM RANGE OP: op should have inputs defined in any possible way: T_args, INT_args, or INPUT variables!");

    helpers::range(*s, *d, *output);

    if (localS)
        delete s;

    if (localD)
        delete d;

    return Status::OK();
}

DECLARE_SHAPE_FN(range) {
    
    const int numInArrs = block.width();
    const int numTArgs  = block.getTArguments()->size();
    const int numIArgs  = block.getIArguments()->size();    

    Nd4jLong steps = 0;
    nd4j::DataType dataType = nd4j::DataType::INHERIT;

    if (numIArgs > 0) {
        Nd4jLong start(0), limit, delta(1);

        if(numIArgs == 1) 
            limit = INT_ARG(0);
        else if(numIArgs == 2) {
            start = INT_ARG(0);
            limit = INT_ARG(1);
        }
        else {
            start = INT_ARG(0);
            limit = INT_ARG(1);
            delta = INT_ARG(2);
        }

        REQUIRE_TRUE(limit != start, 0, "CUSTOM RANGE OP: limit and start values should be different, but got both equal to %f !", limit);
        REQUIRE_TRUE(delta != 0, 0, "CUSTOM RANGE OP: delta should not be equal to zero !");

        if (limit > DataTypeUtils::max<int>())
            dataType = nd4j::DataType::INT64;
        else
            dataType = nd4j::DataType::INT32;

        steps = (limit - start) / delta;

        if(math::nd4j_abs<Nd4jLong>(start + steps * delta) < math::nd4j_abs<Nd4jLong>(limit))
            ++steps;
    }         
    else if (numTArgs > 0) {
        double start(0), limit, delta(1);

        if(numTArgs == 1) 
            limit = T_ARG(0);
        else if(numTArgs == 2) {
            start = T_ARG(0);
            limit = T_ARG(1);
        }
        else {
            start = T_ARG(0);
            limit = T_ARG(1);
            delta = T_ARG(2);
        }

        REQUIRE_TRUE(limit != start, 0, "CUSTOM RANGE OP: limit and start values should be different, but got both equal to %f !", limit);
        REQUIRE_TRUE(delta != 0, 0, "CUSTOM RANGE OP: delta should not be equal to zero !");

        steps = static_cast<Nd4jLong >((limit - start) / delta);
        if (Environment::getInstance()->precisionBoostAllowed())
            dataType = nd4j::DataType::DOUBLE;
        else
            dataType = Environment::getInstance()->defaultFloatDataType();

        if(math::nd4j_abs<double>(start + steps * delta) < math::nd4j_abs<double >(limit))
            ++steps;
    } 
    else if (numInArrs > 0) {
        auto isR = INPUT_VARIABLE(0)->isR();
        auto isZ = INPUT_VARIABLE(0)->isZ();

        if (isR) {
            double start(0), limit, delta(1);

            if (numInArrs == 1)
                limit = INPUT_VARIABLE(0)->e<double>(0);
            else if (numInArrs == 2) {
                start = INPUT_VARIABLE(0)->e<double>(0);
                limit = INPUT_VARIABLE(1)->e<double>(0);
            } else {
                start = INPUT_VARIABLE(0)->e<double>(0);
                limit = INPUT_VARIABLE(1)->e<double>(0);
                delta = INPUT_VARIABLE(2)->e<double>(0);
            }

            REQUIRE_TRUE(limit != start, 0, "CUSTOM RANGE OP: limit and start values should be different, but got both equal to %f !", limit);
            REQUIRE_TRUE(delta != 0, 0, "CUSTOM RANGE OP: delta should not be equal to zero !");

            steps = static_cast<Nd4jLong >((limit - start) / delta);
            dataType = INPUT_VARIABLE(0)->dataType();

            if(math::nd4j_abs<double>(start + steps * delta) < math::nd4j_abs<double >(limit))
                ++steps;
        } else if (isZ) {
            Nd4jLong start(0), limit, delta(1);

            if (numInArrs == 1)
                limit = INPUT_VARIABLE(0)->e<Nd4jLong>(0);
            else if (numInArrs == 2) {
                start = INPUT_VARIABLE(0)->e<Nd4jLong>(0);
                limit = INPUT_VARIABLE(1)->e<Nd4jLong>(0);
            } else {
                start = INPUT_VARIABLE(0)->e<Nd4jLong>(0);
                limit = INPUT_VARIABLE(1)->e<Nd4jLong>(0);
                delta = INPUT_VARIABLE(2)->e<Nd4jLong>(0);
            }

            REQUIRE_TRUE(limit != start, 0, "CUSTOM RANGE OP: limit and start values should be different, but got both equal to %f !", limit);
            REQUIRE_TRUE(delta != 0, 0, "CUSTOM RANGE OP: delta should not be equal to zero !");

            steps = static_cast<Nd4jLong >((limit - start) / delta);
            dataType = INPUT_VARIABLE(0)->dataType();

            if(math::nd4j_abs<double>(start + steps * delta) < math::nd4j_abs<double >(limit))
                ++steps;
        }
    } else {
        REQUIRE_TRUE(false, 0, "CUSTOM RANGE OP: op should have inputs defined in any possible way: T_args, INT_args, or INPUT variables!");
    }

    REQUIRE_TRUE(steps > 0, 0, "CUSTOM RANGE OP: value of (limit-start)/delta should be positive !");

    return SHAPELIST(ShapeBuilders::createVectorShapeInfo(dataType, steps, block.workspace()));
}


}
}

#endif