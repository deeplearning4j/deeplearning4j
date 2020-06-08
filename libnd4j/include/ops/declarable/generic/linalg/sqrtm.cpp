/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sqrtm)
#include <ops/declarable/helpers/sqrtm.h>
#include <ops/declarable/CustomOperations.h>


namespace sd   {
namespace ops  {

CONFIGURABLE_OP_IMPL(sqrtm, 1, 1, false, 0, 0) {

    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(input->rankOf() > 1, 0, "CONFIGURABLE_OP sqrtm: input array rank is required to be > 1, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(input->sizeAt(-2) == input->sizeAt(-1), 0, "CONFIGURABLE_OP sqrtm: two last dimensions of input array should be square matrices, but got such wrong shape instead: %s!", ShapeUtils::shapeAsString(input).c_str());

    helpers::sqrtm(block.launchContext(), input, output);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sqrtm) {
    getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}



}
}

#endif