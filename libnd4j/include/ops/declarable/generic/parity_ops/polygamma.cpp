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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 13.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_polygamma)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/polyGamma.h>

namespace nd4j {
namespace ops  {

CONFIGURABLE_OP_IMPL(polygamma, 2, 1, false, 0, 0) {
    auto n = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);

    auto output   = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(n->isSameShape(x), 0, "POLYGAMMA op: two input arrays n and x must have the same shapes, but got n=%s and x=%s instead !", ShapeUtils::shapeAsString(n).c_str(), ShapeUtils::shapeAsString(x).c_str());

    int arrLen = n->lengthOf();
    for(int i = 0; i < arrLen; ++i ) {
        // TODO case for n == 0 (digamma) should be of OK
        REQUIRE_TRUE((*n)(i) > (T)0., 0, "POLYGAMMA op: all elements of n array must be > 0 !");
        REQUIRE_TRUE((*x)(i) > (T)0., 0, "POLYGAMMA op: all elements of x array must be > 0 !");
    }

    *output = helpers::polyGamma<T>(*n, *x);
    return Status::OK();
}

DECLARE_SYN(polyGamma, polygamma);
DECLARE_SYN(PolyGamma, polygamma);

}
}

#endif