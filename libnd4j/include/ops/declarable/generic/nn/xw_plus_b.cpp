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
 //  xw_plus_b op. Created by GS <george@skymind.io> 31.01.2018
 //  @author Oleg Semeniv <oleg.semeniv@gmail.com>
 //
 //

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_xw_plus_b)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>
#include <helpers/MmulHelper.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(xw_plus_b, 3, 1, false, 0, 0) {

            auto x = INPUT_VARIABLE(0);

            auto b = INPUT_VARIABLE(2);
            auto z = OUTPUT_VARIABLE(0);

            if (x->isEmpty() || INPUT_VARIABLE(1)->isEmpty() || b->isEmpty())
                return Status::OK();

            const bool bTranspose = (block.getIArguments()->size() > 0 ? INT_ARG(0) == 1 : false);

            auto w = bTranspose ? new NDArray(INPUT_VARIABLE(1)->transpose()) : INPUT_VARIABLE(1);

            REQUIRE_TRUE(x->rankOf() == 2, 0, "xw_plus_b: Input x array should have rank equal 2, but got instead %i!", x->rankOf());
            REQUIRE_TRUE(w->rankOf() == 2, 0, "xw_plus_b: Input weights array should have rank equal 2, but got instead %i!", w->rankOf());
            REQUIRE_TRUE(z->rankOf() == 2, 0, "xw_plus_b: Output array should have rank equal 2, but got instead %i!", z->rankOf());

            REQUIRE_TRUE(1 == b->rankOf() && b->lengthOf() == z->sizeAt(-1), 0, "xw_plus_b: Input bias vector should be 1D and have proper dimension 1x%i."
                " But got rank %i, and got length %i instead %i.", z->sizeAt(-1), b->rankOf(), b->lengthOf(), z->sizeAt(-1));

            // multiply x to y
            MmulHelper::mmul(x, w, z, 1.0, 0.0);

            // adding b vector
            z->addiRowVector(*b);

            if (bTranspose)
                delete w;

            return Status::OK();
        }

        DECLARE_SHAPE_FN(xw_plus_b) {

            auto weights = INPUT_VARIABLE(1);

            const int nWeightsFormat = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

            auto weightsShape = (1 == nWeightsFormat) ? ShapeUtils::evalTranspShapeInfo(*weights, block.getWorkspace()) : inputShape->at(1);

            auto outputShape = ShapeUtils::matrixProductShape(inputShape->at(0), weightsShape, false, false,
                ArrayOptions::dataType(inputShape->at(0)), block.getWorkspace());

            return SHAPELIST(CONSTANT(outputShape));
        }

        DECLARE_TYPES(xw_plus_b) {
            getOpDescriptor()
                ->setAllowedInputTypes(sd::DataType::ANY)
                ->setAllowedOutputTypes({ ALL_FLOATS });
        }


        CUSTOM_OP_IMPL(xw_plus_b_bp, 4, 3, false, 0, 0) {

            auto x = INPUT_VARIABLE(0);
            auto b = INPUT_VARIABLE(2);
            auto dLdz = INPUT_VARIABLE(3);

            auto dLdx = OUTPUT_VARIABLE(0);
            auto dLdb = OUTPUT_VARIABLE(2);

            if (x->isEmpty() || INPUT_VARIABLE(1)->isEmpty() || b->isEmpty() || dLdz->isEmpty())
                return Status::OK();

            const bool bTranspose = (block.getIArguments()->size() > 0 ? INT_ARG(0) == 1 : false);

            auto w = bTranspose ? new NDArray(INPUT_VARIABLE(1)->transpose()) : INPUT_VARIABLE(1);

            REQUIRE_TRUE(x->rankOf() == 2, 0, "xw_plus_b BP: Input x array should have rank equal 2, but got instead %i!", x->rankOf());
            REQUIRE_TRUE(w->rankOf() == 2, 0, "xw_plus_b BP: Input weights array should have rank equal 2, but got instead %i!", w->rankOf());
            REQUIRE_TRUE(dLdz->rankOf() == 2, 0, "xw_plus_b BP: Output array should have rank equal 2, but got instead %i!", dLdz->rankOf());
            REQUIRE_TRUE(1 == b->rankOf() && b->lengthOf() == dLdz->sizeAt(-1), 0, "xw_plus_b BP: Input bias vector should be 1D and have proper dimension 1x%i."
                " But got rank %i, and got length %i instead %i.", dLdz->sizeAt(-1), b->rankOf(), b->lengthOf(), dLdz->sizeAt(-1));

            auto dLdw = (bTranspose) ? new NDArray(OUTPUT_VARIABLE(1)->transpose()) : OUTPUT_VARIABLE(1);

            // dLdb
            dLdb->assign(dLdz->reduceAlongDimension(reduce::Sum, { 0 }));

            matmul_bp mmul_bp;
            mmul_bp.execute({ x, w, dLdz }, std::vector<NDArray*>{dLdx, dLdw}, {}, {}, {});

            if (bTranspose) {
                delete w;
                delete dLdw;
            }
            return Status::OK();
        }

        DECLARE_SHAPE_FN(xw_plus_b_bp) {

            Nd4jLong* xShapeInfo;
            Nd4jLong* wShapeInfo;
            Nd4jLong* bShapeInfo;

            COPY_SHAPE(inputShape->at(0), xShapeInfo);
            COPY_SHAPE(inputShape->at(1), wShapeInfo);
            COPY_SHAPE(inputShape->at(2), bShapeInfo);

            return SHAPELIST(CONSTANT(xShapeInfo), CONSTANT(wShapeInfo), CONSTANT(bShapeInfo));
        }

        DECLARE_TYPES(xw_plus_b_bp) {
            getOpDescriptor()
                ->setAllowedInputTypes(sd::DataType::ANY)
                ->setAllowedOutputTypes({ ALL_FLOATS });
        }
    }
}

#endif
