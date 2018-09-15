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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_betainc)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/betaInc.h>

namespace nd4j {
namespace ops  {

CONFIGURABLE_OP_IMPL(betainc, 3, 1, false, 0, 0) {
	auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto x = INPUT_VARIABLE(2);

	auto output   = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(a->isSameShape(b) && a->isSameShape(x), 0, "CONFIGURABLE_OP betainc: all three input arrays must have the same shapes, bit got a=%s, b=%s and x=%s instead !", ShapeUtils::shapeAsString(a).c_str(), ShapeUtils::shapeAsString(b).c_str(), ShapeUtils::shapeAsString(x).c_str());

    int arrLen = a->lengthOf();

    // FIXME: this stuff should be single op call. No sense rolling over couple of arrays twice
    for(int i = 0; i < arrLen; ++i ) {            
        REQUIRE_TRUE(a->getScalar<float>(i) > 0.f,   0, "BETAINC op: arrays a array must contain only elements > 0 !");
        REQUIRE_TRUE(b->getScalar<float>(i) > 0.f,   0, "BETAINC op: arrays b array must contain only elements > 0 !");
        REQUIRE_TRUE(0.f <= x->getScalar<float>(i) && x->getScalar<float>(i) <= 1.f, 0, "BETAINC op: all elements of x array must be within [0, 1] range!");
    }

    *output = helpers::betaInc(*a, *b, *x);

    return Status::OK();
}

DECLARE_SYN(BetaInc, betainc);
DECLARE_SYN(betaInc, betainc);


}
}

#endif