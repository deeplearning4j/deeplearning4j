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

    dnnl::memory::format_tag get_format_tag(const sd::NDArray &array) {
        switch (array.rankOf()) {
            case 1:
                return dnnl::memory::format_tag::ab;
            case 2:
                return array.ordering() == 'c' ? dnnl::memory::format_tag::ab : dnnl::memory::format_tag::ba;
            case 3:
                return array.ordering() == 'c' ? dnnl::memory::format_tag::abc : dnnl::memory::format_tag::cba;
            default:
                throw std::runtime_error("MKLDNN matmul only supports 2D/3D arrays");
        }
    }


//////////////////////////////////////////////////////////////////////////
static void matmulMKLDNN(const NDArray* x, const NDArray* y, NDArray* z, const bool transX, const bool transY, float alpha = 1.f, float beta = 0.f) {

    // mkl works with following
    // [M,K]     x [K,N]     = [M,N]
    // [bS, M,K] x [bS, K,N] = [bS, M,N]

    // possible input cases not supported by mkl, however we'll perform permut/reshape procedures in order to fit requirements
    // [4]          x [4]          = [1]          --> [1,4]     x [4,1]     = [1,1]
    // [4]          x [4,5]        = [5]          --> [1,4]     x [4,5]     = [1,5]
    // [4,5]        x [5]          = [4]          --> [4,5]     x [5,1]     = [4,1]
    // [2,3, 4,5]   x [2,3, 5,4]   = [2,3, 4,4]   --> [6, 4,5]  x [6, 5,4]  = [6, 4,4]
    // [2,2,3, 4,5] x [2,2,3, 5,4] = [2,2,3, 4,4] --> [12, 4,5] x [12, 5,4] = [12, 4,4]

    const auto xRank = x->rankOf();
    const auto yRank = y->rankOf();
    const auto zRank = z->rankOf();

    std::vector<int> permut;

    // fill permutation vector appropriately if transposition is required
    if((transX && xRank > 1) || (transY && yRank > 1)) {

        const int rank = xRank >= yRank ? xRank : yRank;
        permut.resize(rank);
        std::iota(std::begin(permut), std::end(permut), 0);
        permut[rank-2] = rank - 1;
        permut[rank-1] = rank - 2;
    }

    const NDArray* xT = (transX && xRank > 1) ? new NDArray(x->permute(permut)) : x;
    const NDArray* yT = (transY && yRank > 1) ? new NDArray(y->permute(permut)) : y;

    const NDArray* xTR = xRank <= 3 ? xT : new NDArray(xT->reshape(xT->ordering(), {xT->lengthOf() / (xT->sizeAt(-2) * xT->sizeAt(-1)), xT->sizeAt(-2),  xT->sizeAt(-1)}));
    const NDArray* yTR = xRank <= 3 ? yT : new NDArray(yT->reshape(yT->ordering(), {yT->lengthOf() / (yT->sizeAt(-2) * yT->sizeAt(-1)), yT->sizeAt(-2),  yT->sizeAt(-1)}));
          NDArray* zR =  xRank <= 3 ? z  : new NDArray(z->reshape(z->ordering(), {z->lengthOf() / (z->sizeAt(-2) * z->sizeAt(-1)), z->sizeAt(-2),  z->sizeAt(-1)})/*, false*/);

    // [M,K] x [K,N] = [M,N]
    const int64_t M  = (xRank > 1) ? xTR->sizeAt(-2) : 1;
    const int64_t K  = (xRank > 1) ? xTR->sizeAt(-1) : xTR->lengthOf();
    const int64_t N  = (yRank > 1) ? yTR->sizeAt(-1) : 1;
    const int64_t bS = (xRank > 2) ? xTR->sizeAt(0)  : 1;                   // [bS, M,K] x [bS, K,N] = [bS, M,N]

    dnnl::memory::dims xShape = xRank < 3 ? dnnl::memory::dims({M, K}) : dnnl::memory::dims({bS, M, K});
    dnnl::memory::dims yShape = xRank < 3 ? dnnl::memory::dims({K, N}) : dnnl::memory::dims({bS, K, N});
    dnnl::memory::dims zShape = xRank < 3 ? dnnl::memory::dims({M, N}) : dnnl::memory::dims({bS, M, N});

    // x type
    dnnl::memory::data_type xType;
    if(x->dataType() == DataType::FLOAT32)
        xType = dnnl::memory::data_type::f32;
    else if(x->dataType() == DataType::HALF)
        xType = dnnl::memory::data_type::f16;
    else if(x->dataType() == DataType::BFLOAT16)
        xType = dnnl::memory::data_type::bf16;
    else if(x->dataType() == DataType::UINT8)
        xType = dnnl::memory::data_type::u8;
    else
        xType = dnnl::memory::data_type::s8;

    // y type
    dnnl::memory::data_type yType = xType;
    if(y->dataType() == DataType::UINT8)
        yType = dnnl::memory::data_type::u8;
    else if(y->dataType() == DataType::INT8)
        yType = dnnl::memory::data_type::s8;

    // z type
    dnnl::memory::data_type zType = xType;
    if(z->dataType() == DataType::FLOAT32)
        zType = dnnl::memory::data_type::f32;
    else if(z->dataType() == DataType::INT32)
        zType = dnnl::memory::data_type::s32;
    else if(z->dataType() == DataType::UINT8)
        zType = dnnl::memory::data_type::u8;
    else if(z->dataType() == DataType::INT8)
        zType = dnnl::memory::data_type::s8;

    // memory descriptors for arrays

    // x
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xShape, xType, get_format_tag(*xTR));
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, xType, get_format_tag(*xTR));
    if(xTR->ews() != 1) {
        x_user_md.data.format_kind = dnnl_blocked;    // overrides format
        x_user_md.data.format_desc.blocking.strides[0] = xRank == 1 ? 1 : xTR->strideAt(0);
        x_user_md.data.format_desc.blocking.strides[1] = xRank == 1 ? xTR->strideAt(0) : xTR->strideAt(1);
        if(xRank > 2)
            x_user_md.data.format_desc.blocking.strides[2] = xTR->strideAt(2);
    }

    // y
    dnnl::memory::desc y_mkl_md  = dnnl::memory::desc(yShape, yType, get_format_tag(*yTR));
    dnnl::memory::desc y_user_md = dnnl::memory::desc(yShape, yType, get_format_tag(*yTR));
    if(yTR->ews() != 1) {
        y_user_md.data.format_kind = dnnl_blocked;    // overrides format
        y_user_md.data.format_desc.blocking.strides[0] = yRank == 1 ? 1 : yTR->strideAt(0);
        y_user_md.data.format_desc.blocking.strides[1] = yRank == 1 ? yTR->strideAt(0) : yTR->strideAt(1);
        if(yRank > 2)
            y_user_md.data.format_desc.blocking.strides[2] = yTR->strideAt(2);
    }

    // z
    dnnl::memory::desc z_mkl_md  = dnnl::memory::desc(zShape, zType, get_format_tag(*zR));
    dnnl::memory::desc z_user_md = dnnl::memory::desc(zShape, zType, get_format_tag(*zR));
    if(zR->ews() != 1) {
        z_user_md.data.format_kind = dnnl_blocked;    // overrides format
        z_user_md.data.format_desc.blocking.strides[0] = zRank == 1 ? 1 : zR->strideAt(0);
        z_user_md.data.format_desc.blocking.strides[1] = zRank == 1 ? zR->strideAt(0) : zR->strideAt(1);
        if(zRank > 2)
            z_user_md.data.format_desc.blocking.strides[2] = zR->strideAt(2);
    }

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // Create attributes (to handle alpha and beta if necessary)
    dnnl::primitive_attr attr; // it is empty since we have usual values for alpha (=1) and beta (=0)
    if (alpha != 1.f) attr.set_output_scales(0, {alpha});
    if (beta != 0.f) {
        dnnl::post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }

    // operation primitive description
    dnnl::matmul::desc op_desc(x_mkl_md, y_mkl_md, z_mkl_md);
    dnnl::matmul::primitive_desc op_prim_desc(op_desc, attr, engine);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    dnnl::stream stream(engine);

    // provide memory buffers and check whether reorder is required

    // input
    mkldnnUtils::loadDataToMklStream(xTR, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);
    /*
    auto x_user_mem = dnnl::memory(x_user_md, engine, xTR->buffer());
    const bool xReorder = op_prim_desc.src_desc() != x_user_mem.get_desc();
    auto x_mkl_mem = xReorder ? dnnl::memory(op_prim_desc.src_desc(), engine) : x_user_mem;
    if (xReorder)
        dnnl::reorder(x_user_mem, x_mkl_mem).execute(stream, x_user_mem, x_mkl_mem);
    args[DNNL_ARG_SRC] = x_mkl_mem;
*/
    // y
    mkldnnUtils::loadDataToMklStream(yTR, engine, stream, y_user_md, op_prim_desc.weights_desc(), args[DNNL_ARG_WEIGHTS]);
    /*
    auto y_user_mem = dnnl::memory(y_user_md, engine, yTR->buffer());
    const bool yReorder = op_prim_desc.weights_desc() != y_user_mem.get_desc();
    auto y_mkl_mem = yReorder ? dnnl::memory(op_prim_desc.weights_desc(), engine) : y_user_mem;
    if (yReorder)
        dnnl::reorder(y_user_mem, y_mkl_mem).execute(stream, y_user_mem, y_mkl_mem);
    args[DNNL_ARG_WEIGHTS] = y_mkl_mem;
*/
    // z
    auto z_user_mem = dnnl::memory(z_user_md, engine, zR->buffer());
    const bool zReorder = op_prim_desc.dst_desc() != z_user_mem.get_desc();
    auto z_mkl_mem = zReorder ? dnnl::memory(op_prim_desc.dst_desc(), engine) : z_user_mem;
    args[DNNL_ARG_DST] = z_mkl_mem;

    // run calculations
    dnnl::matmul(op_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (zReorder)
        dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

    stream.wait();

    if(zR->buffer() != z->buffer())
        z->assign(zR);

    if(zR != z)
        delete zR;
    if(xTR != xT)
        delete xTR;
    if(xT != x)
        delete xT;
    if(yTR != yT)
        delete yTR;
    if(yT != y)
        delete yT;

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(matmul, ENGINE_CPU) {

    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if(x->isEmpty() || y->isEmpty())
        return Status::OK();

    int iSize = (int) block.getIArguments()->size();
    int transX = iSize > 0 ? INT_ARG(0) : 0;
    int transY = iSize > 1 ? INT_ARG(1) : 0;
    const int transZ = iSize > 2 ? INT_ARG(2) : 0;

    // optional use alpha nad beta
    iSize = (int)block.getTArguments()->size();
    float alpha = iSize > 0 ? T_ARG(0) : 1.0;
    float beta = iSize > 1 ? T_ARG(1) : 0.0;

    const int xRank = x->rankOf();
    const int yRank = y->rankOf();
    const int zRank = z->rankOf();

    if (transZ) {
        x = INPUT_VARIABLE(1);
        y = INPUT_VARIABLE(0);
        bool temp = transX;
        transX = !transY;
        transY = !temp;
    }

    const int xLastDim = transX ? -2 : -1;
    const int yLastDim = transY ? -2 : -1;
    const int xLastButOneDim = transX ? -1 : -2;
    const int yLastButOneDim = transY ? -1 : -2;

    // ******* input validation ******* //
    REQUIRE_TRUE(xRank > 0 && yRank > 0, 0, "MATMUL MKLDNN OP: input arrays must have rank bigger than 0 (should not be scalars), but got instead: x rank = %i, y rank = %i !", xRank, yRank);

    if (xRank == 1 && yRank == 1) {  // dot case, output is scalar (or vector with length = 1)
        REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0,"MATMUL MKLDNN OP: since input arrays are vectors they must have the same length, but got x length = %i, y length = %i !",x->lengthOf(), y->lengthOf());
    } else if (xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
        REQUIRE_TRUE(x->lengthOf() == y->sizeAt(yLastButOneDim), 0, "MATMUL MKLDNN OP: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !", ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
    } else if (xRank == 2 && yRank == 1) {   // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
        REQUIRE_TRUE(x->sizeAt(xLastDim) == y->lengthOf(), 0, "MATMUL MKLDNN OP: input arrays have inconsistent shapes for matrix-vector product: x %s, y %s !", ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
    } else {
        REQUIRE_TRUE(xRank == yRank && yRank == zRank, 0, "MATMUL MKLDNN OP: input and output arrays must have the same rank, but got instead: x rank = %i, y rank = %i, z rank = %i !", xRank, yRank, zRank);
        REQUIRE_TRUE(x->sizeAt(xLastDim) == y->sizeAt(yLastButOneDim) && x->sizeAt(xLastButOneDim) == z->sizeAt(-2) && y->sizeAt(yLastDim) == z->sizeAt(-1), 0, "MATMUL MKLDNN OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !", ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(), ShapeUtils::shapeAsString(z).c_str());

        if (xRank > 2)   // outer dims must be the same
            for (int i = 0; i < xRank - 2; ++i)
                REQUIRE_TRUE(x->sizeAt(i) == y->sizeAt(i) && y->sizeAt(i) == z->sizeAt(i), 0, "MATMUL MKLDNN OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !", ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(), ShapeUtils::shapeAsString(z).c_str());
    }
    // ******* end of input validation ******* //

    matmulMKLDNN(x, y, z, transX, transY, alpha, beta);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(matmul, ENGINE_CPU) {

    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    auto z = OUTPUT_VARIABLE(0);

    const auto xType = x->dataType();
    const auto yType = y->dataType();
    const auto zType = z->dataType();

    float alpha = block.numT() > 0 ? T_ARG(0) : 1.0f;
    float beta = block.numT() > 1 ? T_ARG(1) : 0.0f;

    // we're skipping if result order is F or arrays are not continuous
    bool skip2D = z->rankOf() == 2 && (z->ordering() == 'f' || x->ews() != 1 || y->ews() != 1 || z->ews() != 1);

    // we're skipping 3D cases if they are not C continuoys
    bool skip3D = z->rankOf() == 3 && (x->ordering() == 'f' || y->ordering() == 'f' || z->ordering() == 'f' || x->ews() != 1 || y->ews() != 1 || z->ews() != 1);

    return !skip2D && !skip3D && block.isUseMKLDNN() && x->rankOf() < 3 &&
          (
            (xType==DataType::FLOAT32  && yType==DataType::FLOAT32  && zType==DataType::FLOAT32)  ||
            (xType==DataType::HALF     && yType==DataType::HALF     && zType==DataType::FLOAT32)  ||
            (xType==DataType::BFLOAT16 && yType==DataType::BFLOAT16 && zType==DataType::BFLOAT16) ||
            ((xType==DataType::UINT8 || xType==DataType::INT8) && (yType==DataType::UINT8 || yType==DataType::INT8) && (zType==DataType::UINT8 || zType==DataType::INT8 || zType==DataType::INT32 || zType==DataType::FLOAT32))
          );
}


}
}
}
