//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_CUSTOMOPERATIONS_H
#define LIBND4J_CUSTOMOPERATIONS_H

#include <memory>
#include <op_boilerplate.h>
#include <types/float16.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <Block.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/BooleanOp.h>
#include <ops/declarable/LogicOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/ArrayUtils.h>
#include <ShapeList.h>

namespace nd4j {
    namespace ops {
        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1);

        DECLARE_OP(noop, -1, -1, true);
        DECLARE_OP(testop2i2o, 2, 2, true);
        DECLARE_OP(softmax, 1, 1, true);
        DECLARE_OP(softmax_bp, 2, 1, true);
        DECLARE_OP(biasadd, 2, 1, true);
        DECLARE_OP(floor, 1, 1, true);
        DECLARE_OP(realdiv, 2, 1, true);
        DECLARE_OP(merge, -1, 1, true);         // should become custom
        DECLARE_OP(broadcastgradientargs, 2, 2, true);
        DECLARE_OP(assign, 2, 1, false);
        DECLARE_OP(mergemax, -1, 1, false);
        DECLARE_OP(mergemaxindex, -1, 1, false);
        DECLARE_OP(mergeadd, -1, 1, false);
        DECLARE_OP(mergeavg, -1, 1, false);
        DECLARE_OP(identity, 1, 1, true);
        DECLARE_OP(add, 2, 1, true);
        DECLARE_OP(subtract, 2, 1, true);
        DECLARE_OP(reversesubtract, 2, 1, true);
        DECLARE_OP(multiply, 2, 1, true);
        DECLARE_OP(divide, 2, 1, true);
        DECLARE_OP(reversedivide, 2, 1, true);
        DECLARE_OP(reshapeas, 2, 1, true);      // should become custom
        DECLARE_OP(transpose, 1, 1, true);      // should become custom
        DECLARE_OP(zeros_as, 1, 1, false);
        DECLARE_OP(maximum, 2, 1, true);
        DECLARE_OP(minimum, 2, 1, true);


        DECLARE_DIVERGENT_OP(Switch, 2, 2, true);

        DECLARE_LOGIC_OP(While);
        DECLARE_LOGIC_OP(Scope);
        DECLARE_LOGIC_OP(Conditional);

        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1);
        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);
        DECLARE_CUSTOM_OP(conv2d, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(conv2d_bp, 3, 2, false, 0, 9);
        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0);
        DECLARE_CUSTOM_OP(reshape, 1, 1, true, 0, -1);
        DECLARE_CUSTOM_OP(sconv2d, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(sconv2d_bp, 4, 2, false, 0, 9);
        DECLARE_CUSTOM_OP(deconv2d, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(deconv2d_bp, 4, 2, false, 0, 9);
        DECLARE_CUSTOM_OP(maxpool2d, 1, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(avgpool2d, 1, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(pnormpool2d, 1, 1, false, 0, 10);
        DECLARE_CUSTOM_OP(maxpool3d_bp, 3, 1, true, 0, 13);
        DECLARE_CUSTOM_OP(avgpool3d, 1, 1, true, 0, 11);
        DECLARE_CUSTOM_OP(avgpool3d_bp, 2, 1, true, 0, 11);
        DECLARE_CUSTOM_OP(fullconv3d, 5, 1, false, 0, 13);
        DECLARE_CUSTOM_OP(fullconv3d_bp, 5, 1, false, 0, 13);
        DECLARE_CUSTOM_OP(fullconv3d_grad, 4, 2, false, 1, 13);
        DECLARE_CUSTOM_OP(maxpool2d_bp, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(pooling2d, 1, 1, false, 0, 11);
        DECLARE_CUSTOM_OP(avgpool2d_bp, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(pnormpool2d_bp, 2, 1, false, 1, 10);
        DECLARE_CUSTOM_OP(tear, 1, -1, false, 0, -1);
        DECLARE_CUSTOM_OP(unstack, 1, -1, false, 0, 1);
        DECLARE_CUSTOM_OP(im2col, 1, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(col2im, 1, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(strided_slice, 1, 1, true, 0, -1); // TODO: new op type needed. that returns VIEW
        DECLARE_CUSTOM_OP(upsampling2d, 1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(upsampling2d_bp, 2, 1, false, 0, 1);


        // recurrent ops
        DECLARE_CUSTOM_OP(sru1, 5, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(sru2, 5, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(sru_bp_1, 8, 4, true, 0, 0);
        DECLARE_CUSTOM_OP(sru_bp_2, 8, 4, true, 0, 0);


        DECLARE_CONFIGURABLE_OP(tensormmul, 2, 1, false, 0, -1);   // should become custom
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1);
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(repeat, 1, 1, true, 0, -1);   // should become custom
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(permute, 1, 1, true, 0, -1);   // MAYBE should become custom :/
        DECLARE_CONFIGURABLE_OP(sum, 1, 1, false, 0, -1);       // should become reduction
        DECLARE_CONFIGURABLE_OP(batchnorm, 1, 1, true, 4, 3);
        DECLARE_CONFIGURABLE_OP(batchnorm_bp, 5, 1, true, 0, 1);        
        DECLARE_CONFIGURABLE_OP(conv3d, 2, 1, false, 0, 7); // make this custom
        DECLARE_CONFIGURABLE_OP(conv3d_bp, 3, 1, false, 0, 7); // TODO: to be implemented

        DECLARE_CONFIGURABLE_OP(maxpool3d, 1, 2, true, 0, 13); // make this one custom
        DECLARE_CONFIGURABLE_OP(ismax, 1, 1, false, 0, -1);


        DECLARE_CONFIGURABLE_OP(firas_sparse, 1, 1, false, 0, -1);



        DECLARE_BOOLEAN_OP(lt_scalar, 2, true);
        DECLARE_BOOLEAN_OP(gt_scalar, 2, true);
        DECLARE_BOOLEAN_OP(lte_scalar, 2, true);
        DECLARE_BOOLEAN_OP(gte_scalar, 2, true);
        DECLARE_BOOLEAN_OP(eq_scalar, 2, true);
        DECLARE_BOOLEAN_OP(neq_scalar, 2, true);
    }
}

#endif //LIBND4J_CUSTOMOPERATIONS_H
