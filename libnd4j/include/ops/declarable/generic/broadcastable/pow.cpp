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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_Pow)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
        BROADCASTABLE_OP_IMPL(Pow, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            BROADCAST_CHECK_EMPTY(x,y,z);

            //REQUIRE_TRUE(!y->isB(), 0, "Pairwise OP: you can't divide by bool array!");

            auto tZ = BroadcastHelper::broadcastApply({scalar::Pow, pairwise::Pow, broadcast::Pow}, x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            return Status::OK();
        }

        DECLARE_TYPES(Pow) {
            getOpDescriptor()
                ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
                ->setAllowedInputTypes(1, {ALL_FLOATS, ALL_INTS})
                ->setAllowedOutputTypes(0, {ALL_FLOATS, ALL_INTS});
        }

       CUSTOM_OP_IMPL(Pow_bp, 3, 2, false, 0, 0) {

           auto x = INPUT_VARIABLE(0);
           auto y = INPUT_VARIABLE(1);
           auto dLdz = INPUT_VARIABLE(2);
       
           auto dLdx = OUTPUT_VARIABLE(0);
           auto dLdy = OUTPUT_VARIABLE(1);
       
           const Nd4jLong* dLdzShapeInfo = nullptr;
           const bool areShapesBroadcastable = ShapeUtils::evalBroadcastShapeInfo(x->shapeInfo(), y->shapeInfo(), true, dLdzShapeInfo, block.getWorkspace());
           REQUIRE_TRUE(areShapesBroadcastable, 0, "POW_BP OP: the shapes of x %s"
               " and y %s are not suitable for broadcast !", 
               ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());
           REQUIRE_TRUE(shape::equalsSoft(dLdz->shapeInfo(), dLdzShapeInfo), 0, 
               "POW_BP OP: wrong shape of next epsilon array (dLdOut),"
               " expected is %s, but got %s instead !", 
               ShapeUtils::shapeAsString(dLdzShapeInfo).c_str(), ShapeUtils::shapeAsString(dLdz).c_str());
       
           // dL/dy = x^y * log(x) * dL/dz
           auto temp = x->applyTrueBroadcast(BroadcastOpsTuple::Pow(), *y); // a = x^y
           x->applyTransform(transform::Log, *dLdx); // b = log(x)
           dLdx->applyScalar(sd::scalar::ReplaceNans, 0, *dLdx);
           temp *= *dLdx; // c = b*a
           temp *= *dLdz; // dL/dy = c * dL/dz
           if (dLdy->isSameShape(*dLdz)) {
               dLdy->assign(temp); 
           }
           else {
               std::vector<int> axesForY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), dLdz->shapeInfo());
               dLdy->assign(temp.reduceAlongDimension(reduce::Sum, axesForY)); // dL/dy = sum(c * dL/dz)
           }
           
           // dL/dx = y*x^(y-1) * dL/dz 
           x->applyTrueBroadcast(BroadcastOpsTuple::PowDerivative(), *y, temp); // a = y*x^(y-1)
           temp *= *dLdz; // dLdx = a*dL/dz

           if (dLdx->isSameShape(*dLdz)) {
               dLdx->assign(temp); // dLdx = a*dL/dz
           }
           else {
               std::vector<int> axesForX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), dLdz->shapeInfo());
               dLdx->assign(temp.reduceAlongDimension(reduce::Sum, axesForX)); // dLdx = a*dL/dz
           }
       
           return Status::OK();
       }
       
       DECLARE_SHAPE_FN(Pow_bp) {
       
           auto xShapeInfo = inputShape->at(0);
           auto yShapeInfo = inputShape->at(1);
       
           Nd4jLong* dLdxShapeInfo = nullptr;
           Nd4jLong* dLdyShapeInfo = nullptr;
       
           COPY_SHAPE(xShapeInfo, dLdxShapeInfo);
           COPY_SHAPE(yShapeInfo, dLdyShapeInfo);
       
           return SHAPELIST(CONSTANT(dLdxShapeInfo), CONSTANT(dLdyShapeInfo));
       }
       
       DECLARE_TYPES(Pow_bp) {
           getOpDescriptor()
               ->setAllowedInputTypes({ ALL_FLOATS, ALL_INTS })
               ->setAllowedOutputTypes({ ALL_FLOATS }); 
       }

}
}

#endif