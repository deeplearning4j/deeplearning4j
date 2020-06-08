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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_tensormmul)

#include <numeric>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/MmulHelper.h>


namespace sd {
namespace ops  {

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tensormmul, 2, 1, false, 0, -1) {

    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    auto c = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(a->dataType() == b->dataType(), 0, "tensormmul: A, B and C data types must be the same");

    // building axes
    int axe0_size = INT_ARG(0);
    int axe1_size = INT_ARG(axe0_size+1);
    std::vector<int> axes_0(axe0_size), axes_1(axe1_size);
    for (int e = 0; e < axe0_size; e++)
        axes_0[e] = (int)INT_ARG(e + 1);

    for (int e = 0; e < axe1_size; e++)
        axes_1[e] = (int)INT_ARG(e + axe0_size + 2);

    nd4j_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());

    MmulHelper::tensorDot(a, b, c, axes_0, axes_1);
    return Status::OK();
}
DECLARE_SYN(tensordot, tensormmul);

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(tensormmul) {

    auto aShapeInfo = inputShape->at(0);
    auto bShapeInfo = inputShape->at(1);

    REQUIRE_TRUE(ArrayOptions::dataType(aShapeInfo) == ArrayOptions::dataType(bShapeInfo), 0, "tensormmul: A and B data types must be the same");

    // building axes
    int axe0_size = INT_ARG(0);
    int axe1_size = INT_ARG(axe0_size+1);
    std::vector<int> axes_0(axe0_size), axes_1(axe1_size);
    for (int e = 0; e < axe0_size; e++)
        axes_0[e] = (int) INT_ARG(e+1);

    for (int e = 0; e < axe1_size; e++)
        axes_1[e] = (int) INT_ARG(e + axe0_size + 2);

    // evaluate shapes
    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;
    auto outShape = sd::ShapeUtils::evalShapeForTensorDot(aShapeInfo, bShapeInfo, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(ArrayOptions::dataType(aShapeInfo), 'c', outShape)));
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(tensormmul) {
    getOpDescriptor()
            ->setAllowedInputTypes(0, {DataType::FLOAT32, DataType ::DOUBLE, DataType::HALF})
            ->setAllowedInputTypes(1, {DataType::FLOAT32, DataType ::DOUBLE, DataType::HALF})
            ->setAllowedOutputTypes(0, {DataType::FLOAT32, DataType ::DOUBLE, DataType::HALF});
}

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tensormmul_bp, 3, 2, false, 0, -1) {

    auto A = INPUT_VARIABLE(0);
    auto B = INPUT_VARIABLE(1);

    auto dLdC = INPUT_VARIABLE(2);

    auto dLdA = OUTPUT_VARIABLE(0);
    auto dLdB = OUTPUT_VARIABLE(1);

    REQUIRE_TRUE( (A->dataType() == B->dataType() && (dLdC->dataType() == A->dataType())), 0, "tensormmul_bp: A, B and dLdC data types must be the same");

    int axe0Size = INT_ARG(0);
    int axe1Size = INT_ARG(axe0Size + 1);

    auto Arank = A->rankOf();
    auto Brank = B->rankOf();
    auto dLdCrank = dLdC->rankOf();

    REQUIRE_TRUE((Arank >= axe0Size), 0, "tensormmul_bp: A rank must be the higher or same as input axes 0");

    REQUIRE_TRUE((Brank >= axe1Size), 0, "tensormmul_bp: B rank must be the higher or same as input axes 1");

    // building axes
    std::vector<int> axes0(axe0Size), axes1(axe1Size);
    for (uint e = 0; e < axe0Size; e++)
        axes0[e] = (int)INT_ARG(e + 1);
    for (uint e = 0; e < axe1Size; e++)
        axes1[e] = (int)INT_ARG(e + axe0Size + 2);

    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;

    ShapeUtils::evalShapeForTensorDot(A, B, axes0, axes1, permutAt, permutBt, shapeAt, shapeBt);

    // special case for scalar value
    if (dLdC->isScalar()) {

        dLdA->assign((*dLdC) * *B);
        dLdB->assign((*dLdC) * *A);

        return Status::OK();
    }

    std::vector<int> axesA = ShapeUtils::evalDimsToExclude(Arank, axes0);
    std::vector<int> axesB = ShapeUtils::evalDimsToExclude(Brank, axes1);

    // rank always have to be divided by 2
    std::vector<int> axesAdLdC, axesBdLdC;
    if (dLdCrank > 1) {
        axesAdLdC.resize(dLdCrank / 2);
        std::iota(axesAdLdC.begin(), axesAdLdC.end(), 0);
        axesBdLdC = ShapeUtils::evalDimsToExclude(dLdCrank, axesAdLdC);
    }
    else {
        axesAdLdC.push_back(0);
        axesBdLdC.push_back(0);
    }

    // calculate dLdA
    MmulHelper::tensorDot(dLdC, B, dLdA, axesBdLdC, axesB, permutAt);

    // calculate dLdB
    MmulHelper::tensorDot(A, dLdC, dLdB, axesA, axesAdLdC, permutBt);

    return Status::OK();
}

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(tensormmul_bp) {

    auto aShapeInfo = inputShape->at(0);
    auto bShapeInfo = inputShape->at(1);
    auto dLShapeInfo = inputShape->at(2);

    REQUIRE_TRUE((ArrayOptions::dataType(aShapeInfo) == ArrayOptions::dataType(bShapeInfo) &&
                 (ArrayOptions::dataType(dLShapeInfo) == ArrayOptions::dataType(aShapeInfo))), 0, "tensormmul_bp: A, B and dLdC data types must be the same");

    Nd4jLong* dLdAShapeInfo = nullptr;
    Nd4jLong* dLdBShapeInfo = nullptr;

    COPY_SHAPE(aShapeInfo, dLdAShapeInfo);
    COPY_SHAPE(bShapeInfo, dLdBShapeInfo);

    return SHAPELIST(CONSTANT(dLdAShapeInfo), CONSTANT(dLdBShapeInfo));
}

////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(tensormmul_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, { DataType::FLOAT32, DataType::DOUBLE, DataType::HALF }) // maybe better ALL_FLOATS
        ->setAllowedInputTypes(1, { DataType::FLOAT32, DataType::DOUBLE, DataType::HALF })
        ->setAllowedInputTypes(2, { DataType::FLOAT32, DataType::DOUBLE, DataType::HALF })
        ->setAllowedOutputTypes(0, { DataType::FLOAT32, DataType::DOUBLE, DataType::HALF })
        ->setAllowedOutputTypes(1, { DataType::FLOAT32, DataType::DOUBLE, DataType::HALF });
}
}
}

#endif