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
// @author raver119@gmail.com, created on 07.10.2017.
// @author GS <sgazeos@gmail.com>, modified
// @author Yurii Shyrma (iuriish@yahoo.com), fully rewritten
//

#include <op_boilerplate.h> 
#if NOT_EXCLUDED(OP_matmul)

#include <ops/declarable/CustomOperations.h>
#include <MmulHelper.h>

namespace nd4j {
    namespace ops {

        CUSTOM_OP_IMPL(matmul, 2, 1, false, 0, -2) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            const int iSize = (int) block.getIArguments()->size();
            int transX = iSize > 0 ? INT_ARG(0) : 0;
            int transY = iSize > 1 ? INT_ARG(1) : 0;
            const int transZ = iSize > 2 ? INT_ARG(2) : 0;

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
            REQUIRE_TRUE(xRank > 0 && yRank > 0, 0,
                         "MATMUL OP: input arrays must have rank bigger than 0 (should not be scalars), but got instead: x rank = %i, y rank = %i !",
                         xRank, yRank);

            if (xRank == 1 && yRank == 1) {  // dot case, output is scalar (or vector with length = 1)
                REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0,
                             "MATMUL OP: since input arrays are vectors they must have the same length, but got x length = %i, y length = %i !",
                             x->lengthOf(), y->lengthOf());
            } else if (xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
                REQUIRE_TRUE(x->lengthOf() == y->sizeAt(yLastButOneDim), 0,
                             "MATMUL OP: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !",
                             ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
            } else if (xRank == 2 && yRank == 1) {   // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
                REQUIRE_TRUE(x->sizeAt(xLastDim) == y->lengthOf(), 0,
                             "MATMUL OP: input arrays have inconsistent shapes for matrix-vector product: x %s, y %s !",
                             ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
            } else {
                REQUIRE_TRUE(xRank == yRank && yRank == zRank, 0,
                             "MATMUL OP: input and output arrays must have the same rank, but got instead: x rank = %i, y rank = %i, z rank = %i !",
                             xRank, yRank, zRank);
                REQUIRE_TRUE(x->sizeAt(xLastDim) == y->sizeAt(yLastButOneDim) &&
                             x->sizeAt(xLastButOneDim) == z->sizeAt(-2) && y->sizeAt(yLastDim) == z->sizeAt(-1), 0,
                             "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                             ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                             ShapeUtils::shapeAsString(z).c_str());

                if (xRank > 2)   // outer dims must be the same
                    for (int i = 0; i < xRank - 2; ++i)
                        REQUIRE_TRUE(x->sizeAt(i) == y->sizeAt(i) && y->sizeAt(i) == z->sizeAt(i), 0,
                                     "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !",
                                     ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str(),
                                     ShapeUtils::shapeAsString(z).c_str());
            }
            // ******* end of input validation ******* //

            MmulHelper::matmul(x, y, z, transX, transY);

            return Status::OK();
        }

        DECLARE_SYN(mMul, matmul);

        DECLARE_SYN(mmul, matmul);

        DECLARE_SYN(gemm, matmul);

        DECLARE_SYN(gemv, matmul);

        DECLARE_SYN(dot, matmul);


        DECLARE_SHAPE_FN(matmul) {

            auto xShapeInfo = inputShape->at(0);
            auto yShapeInfo = inputShape->at(1);

            const int iSize = (int) block.getIArguments()->size();
            int transX = iSize > 0 ? INT_ARG(0) : 0;
            int transY = iSize > 1 ? INT_ARG(1) : 0;
            const int transZ = iSize > 2 ? INT_ARG(2) : 0;

            REQUIRE_TRUE(xShapeInfo[0] > 0 && yShapeInfo[0] > 0, 0,
                         "MATMUL OP: input arrays must have rank bigger than 0 (should not be scalars), but got instead: x rank = %i, y rank = %i !",
                         xShapeInfo[0], yShapeInfo[0]);

            if (transZ) {
                xShapeInfo = inputShape->at(1);
                yShapeInfo = inputShape->at(0);
                bool temp = transX;
                transX = !transY;
                transY = !temp;
            }

            auto zShapeOnly = ShapeUtils::evalShapeForMatmul(xShapeInfo, yShapeInfo, transX, transY);

            auto dtypeX = ArrayOptions::dataType(xShapeInfo);
            auto dtypeY = ArrayOptions::dataType(yShapeInfo);

            auto xOrder = shape::order(xShapeInfo);
            auto yOrder = shape::order(yShapeInfo);
            auto zOrder = xOrder == 'c' && yOrder == 'c' ? 'c' : 'f';

            // we just pick the higher data type out of X and Y
            auto dtypeZ = dtypeX > dtypeY ? dtypeX : dtypeY;

            auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(dtypeZ, zOrder, zShapeOnly);
            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(matmul) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS});
        }


        CUSTOM_OP_IMPL(matmul_bp, 3, 2, false, 0, -2) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto eps = INPUT_VARIABLE(2);
            auto dldx = OUTPUT_VARIABLE(0);
            auto dldy = OUTPUT_VARIABLE(1);

            const int iSize = (int) block.getIArguments()->size();
            int transX = iSize > 0 ? INT_ARG(0) : 0;
            int transY = iSize > 1 ? INT_ARG(1) : 0;
            const int transZ = iSize > 2 ? INT_ARG(2) : 0;

/*
In: x=[a,b], y=[b,c]
tX  tY  tZ  x       y       z       dz          dLdx                                    dLdy
F   F   F   [a,b]   [b,c]   [a,c]   [a,c]       [a,c]*[b,c]T = [a,b]        x*yT        [a,b]T*[a,c] = [b,c]        xT*y
T   F   F   [b,a]   [b,c]   [a,c]   [a,c]       ([a,c]*[b,c]T)T = [b,a]     (x*yT)T     [b,a]*[a,c] = [b,c]         x*y
F   T   F   [a,b]   [c,b]   [a,c]   [a,c]       ([a,c]*[c,b]) = [a,b]       x*y         [a,b]T*[a,c] = [b,c] ->T    xT*y
T   T   F   [b,a]   [c,b]   [a,c]   [a,c]       ([a,c]*[c,b])T = [b,a]      (x*y)T      [b,a]*[a,c] = [b,c]  ->T    x*y
F   F   T   [a,b]   [b,c]   [c,a]   [c,a]
*/


            nd4j::ops::matmul op;
            op.execute({eps, y}, {dldx}, {}, {transZ, !transY, transX}, {});
            op.execute({x, eps}, {dldy}, {}, {!transX, transZ, transY}, {});

            return Status::OK();
        }


        DECLARE_SHAPE_FN(matmul_bp) {
            Nd4jLong *xShapeInfo;
            Nd4jLong *yShapeInfo;

            COPY_SHAPE(inputShape->at(0), xShapeInfo);
            COPY_SHAPE(inputShape->at(1), yShapeInfo);

            return SHAPELIST(xShapeInfo, yShapeInfo);
        }

        DECLARE_TYPES(matmul_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedInputTypes(2, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(1, {ALL_FLOATS});
        }

    }
}


#endif