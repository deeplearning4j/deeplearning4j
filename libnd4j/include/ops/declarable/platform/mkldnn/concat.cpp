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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <numeric>


namespace sd      {
namespace ops       {
namespace platforms {


//////////////////////////////////////////////////////////////////////////
static void concatMKLDNN(const std::vector<const NDArray*>& inArrs, NDArray& output, const int axis) {

    // data type
    dnnl::memory::data_type type;
    if(output.dataType() == DataType::FLOAT32)
        type = dnnl::memory::data_type::f32;
    else if(output.dataType() == DataType::HALF)
        type = dnnl::memory::data_type::f16;
    else if(output.dataType() == DataType::BFLOAT16)
        type = dnnl::memory::data_type::bf16;
    else if(output.dataType() == DataType::UINT8)
        type = dnnl::memory::data_type::u8;
    else
        type = dnnl::memory::data_type::s8;

    std::vector<dnnl::memory::desc> x_user_md(inArrs.size()), x_mkl_md(inArrs.size());

    // inputs
    for (int i = 0; i < inArrs.size(); ++i) {

        dnnl::memory::dims dims = inArrs[i]->getShapeAsFlatVector();
        x_user_md[i] = x_mkl_md[i] = dnnl::memory::desc(dims, type, mkldnnUtils::getFormat(*inArrs[i]));
        mkldnnUtils::setBlockStrides(*inArrs[i], x_user_md[i]);
    }

    // output
    dnnl::memory::dims dims = output.getShapeAsFlatVector();
    dnnl::memory::desc z_mkl_md = dnnl::memory::desc(dims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc z_user_md = dnnl::memory::desc(dims, type, mkldnnUtils::getFormat(output));
    mkldnnUtils::setBlockStrides(output, z_user_md);

    std::unordered_map<int, dnnl::memory> args;

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    dnnl::concat::primitive_desc op_prim_desc(axis, x_mkl_md, engine);

    dnnl::stream stream(engine);

    // inputs
    for (int i = 0; i < inArrs.size(); ++i)
        mkldnnUtils::loadDataToMklStream(*inArrs[i], engine, stream, x_user_md[i], op_prim_desc.src_desc(i), args[DNNL_ARG_MULTIPLE_SRC + i]);

    // outputs
    auto z_user_mem = mkldnnUtils::loadDataToMklStream(output, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

    // primitive execution
    dnnl::concat(op_prim_desc).execute(stream, args);

    // reorder output if necessary
    if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
        dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

    stream.wait();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(concat, ENGINE_CPU) {

    REQUIRE_TRUE(block.width() > 0, 0, "CONCAT MKLDNN op: No input arrays were provided");

    const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

    const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

    // first of all take into account possible presence of empty arrays
    // also if scalar is present -> copy its value to vector with length=1
    std::vector<const NDArray*> nonEmptyArrs;
    std::vector<int> arrsToDelete;
    int index = 0;
    bool allOfSameType = true;
    auto rankOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->rankOf() : 0;
    auto typeOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->dataType() : block.dataType();

    for(int i = 0; i < numOfInArrs; ++i) {
        auto input = INPUT_VARIABLE(i);
        auto currentRank = input->rankOf();

        if(!input->isEmpty()) {

            allOfSameType &= (typeOfFirstArr == input->dataType());

            if(input->rankOf() == 0) {
                auto vec = new NDArray('c', {1}, input->dataType(), block.launchContext());
                vec->assign(input);
                nonEmptyArrs.push_back(vec);
                arrsToDelete.push_back(index);
            }
            else{
                nonEmptyArrs.push_back(input);
            }
            ++index;
        }
    }

    const int numOfNonEmptyArrs = nonEmptyArrs.size();

    if(numOfNonEmptyArrs == 0){
        //All inputs are empty arrays -> return empty, mainly for TF import compatibility (no op)
        REQUIRE_TRUE(OUTPUT_VARIABLE(0)->isEmpty(), 0, "CONCAT MKLDNN op: If all input variables are empty, output must be empty");
        return Status::OK();
    }

    const int rank = nonEmptyArrs[0]->rankOf();                     //  look up to first non-empty array
    int axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0) : INT_ARG(0);
    if(axis < 0){
        axis += rank;
    }

    // ******** input validation ******** //
    REQUIRE_TRUE(allOfSameType, 0, "CONCAT MKLDNN op: all of input arrays must have same type !");
    REQUIRE_TRUE(nonEmptyArrs[0]->dataType() == OUTPUT_VARIABLE(0)->dataType(), 0, "CONCAT MKLDNN op: output array should have the same type as inputs arrays !");
    REQUIRE_TRUE(0 <= axis && (axis < rank || (axis == 0 && rank == 0)), 0, "CONCAT MKLDNN op: input axis must be in range [0, %i], but got %i instead!", rank-1, axis);

    for(int i = 1; i < numOfNonEmptyArrs; ++i)
        REQUIRE_TRUE(nonEmptyArrs[i]->rankOf() == rank, 0, "CONCAT MKLDNN op: all input arrays must have the same rank !");

    for(int i = 1; i < numOfNonEmptyArrs; ++i) {
        for(int dim = 0; dim < rank; ++dim)
            if(dim != axis)
                REQUIRE_TRUE(nonEmptyArrs[i]->sizeAt(dim) == nonEmptyArrs[0]->sizeAt(dim), 0, "CONCAT MKLDNN op: all input arrays must have the same dimensions (except those on input axis) !");
    }
    // ******** end of input validation ******** //

    auto output = OUTPUT_VARIABLE(0);

    if(numOfNonEmptyArrs == 1)
        output->assign(nonEmptyArrs[0]);
    else
        concatMKLDNN(nonEmptyArrs, *output, axis);

    // delete dynamically allocated vectors with length=1
    for(int index : arrsToDelete)
        delete nonEmptyArrs[index];

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(concat, ENGINE_CPU) {

    auto z = OUTPUT_VARIABLE(0);

    const auto zType = z->dataType();

    const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);
    const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

    return z->rankOf() < 7 && numOfInArrs <= 3072
           && (zType==DataType::FLOAT32 || zType==DataType::HALF || zType==DataType::BFLOAT16 || zType==DataType::UINT8 || zType==DataType::INT8);
}

}
}
}
