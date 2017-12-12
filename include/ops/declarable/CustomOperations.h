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
#include <Context.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/BooleanOp.h>
#include <ops/declarable/LogicOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>
#include <ops/declarable/DeclarableListOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/ArrayUtils.h>
#include <helpers/ShapeUtils.h>
#include <array/ShapeList.h>

namespace nd4j {
    namespace ops {
        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1);
        DECLARE_REDUCTION_OP(argmax, 1, 1, false, 0, -2);
        DECLARE_REDUCTION_OP(argmin, 1, 1, false, 0, -2);
        DECLARE_REDUCTION_OP(norm, 1, 1, false, 1, -2);

        DECLARE_OP(noop, -1, -1, true);
        DECLARE_OP(testop2i2o, 2, 2, true);
        DECLARE_OP(softmax, 1, 1, true);
        DECLARE_OP(softmax_bp, 2, 1, true);
        DECLARE_OP(biasadd, 2, 1, true);
        DECLARE_OP(floor, 1, 1, true);
        DECLARE_OP(merge, -1, 1, true);         // should become custom
        DECLARE_OP(broadcastgradientargs, 2, 2, true);
        DECLARE_OP(assign, 2, 1, false);
        DECLARE_OP(mergemax, -1, 1, false);
        DECLARE_OP(mergemaxindex, -1, 1, false);
        DECLARE_OP(mergeadd, -1, 1, false);
        DECLARE_OP(mergeavg, -1, 1, false);
        DECLARE_OP(identity, 1, 1, true);
        DECLARE_OP(identity_bp, 2, 1, true);
        DECLARE_OP(zeros_as, 1, 1, false);
        DECLARE_OP(ones_as, 1, 1, false);
        DECLARE_OP(square, 1, 1, true);
        DECLARE_OP(equals, 2, 1, true);
        DECLARE_OP(not_equals, 2, 1, true);
        DECLARE_OP(less_equal, 2, 1, true);
        DECLARE_OP(greater_equal, 2, 1, true);
        DECLARE_OP(less, 2, 1, true);
        DECLARE_OP(greater, 2, 1, true);
        DECLARE_OP(log1p, 2, 1, true);
        DECLARE_OP(toggle_bits, -1, -1, true);
        DECLARE_OP(rint, 1, 1, true);
        DECLARE_OP(listdiff, 2, 2, false);

        DECLARE_OP(to_double, 1, 1, true);
        DECLARE_OP(to_float16, 1, 1, true);
        DECLARE_OP(to_float32, 1, 1, true);
        DECLARE_OP(to_int32, 1, 1, true);
        DECLARE_OP(to_int64, 1, 1, true);
        DECLARE_OP(to_uint32, 1, 1, true);
        DECLARE_OP(to_uint64, 1, 1, true);

        DECLARE_OP(scatter_add, 3, 1, true);
        DECLARE_OP(scatter_sub, 3, 1, true);
        DECLARE_OP(scatter_mul, 3, 1, true);
        DECLARE_OP(scatter_div, 3, 1, true);
        DECLARE_OP(scatter_upd, 3, 1, true);
    


        DECLARE_DIVERGENT_OP(Switch, 2, 2, true);

        DECLARE_LOGIC_OP(While);
        DECLARE_LOGIC_OP(Scope);
        DECLARE_LOGIC_OP(Conditional);
        DECLARE_LOGIC_OP(Return);

        // TODO: make broadcastables separate class
        DECLARE_CUSTOM_OP(maximum, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(minimum, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(add, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(subtract, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(reversesubtract, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(reversemod, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(squaredsubtract, 2, 1, true, 0, 0)
        DECLARE_CUSTOM_OP(multiply, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(divide, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(reversedivide, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(floormod, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(floordiv, 2, 1, true, 0, 0)
        DECLARE_CUSTOM_OP(realdiv, 2, 1, true, 0, 0);

        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1);
        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(concat_bp, -1, -1, false, 0, 1);
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0);
        DECLARE_CUSTOM_OP(conv1d, 2, 1, false, 0, 3);
        DECLARE_CUSTOM_OP(conv1d_bp, 3, 2, false, 0, 3);
        DECLARE_CUSTOM_OP(conv2d, 2, 1, false, 0, 3);
        DECLARE_CUSTOM_OP(conv2d_bp, 3, 2, false, 0, 9);
        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0);
        DECLARE_CUSTOM_OP(reshape, 1, 1, true, 0, -2);
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
        DECLARE_CUSTOM_OP(strided_slice, 1, 1, false, 0, 5); // TODO: new op type needed. that returns VIEW
        DECLARE_CUSTOM_OP(slice, 1, 1, false, 0, -1);
        DECLARE_CUSTOM_OP(upsampling2d, 1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(upsampling2d_bp, 2, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(tensormmul, 2, 1, false, 0, -1);   
        DECLARE_CUSTOM_OP(repeat, 1, 1, true, 0, -1);   
        DECLARE_CUSTOM_OP(conv3d, 2, 1, false, 0, 7); 
        DECLARE_CUSTOM_OP(maxpool3d, 1, 2, true, 0, 13); 
        DECLARE_CUSTOM_OP(permute, 1, 1, true, 0, -2);   
        DECLARE_CUSTOM_OP(reshapeas, 2, 1, true, 0, 0);      
        DECLARE_CUSTOM_OP(transpose, 1, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(stack, -1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(size, 1, 1, false, 0, 0); // add DeclarableScalarOp?
        DECLARE_CUSTOM_OP(rank, 1, 1, false, 0, 0); // ^^^^
        DECLARE_CUSTOM_OP(onehot, 1, 1, false, 2, 2);
        DECLARE_CUSTOM_OP(expand_dims, 1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(range, -2, 1, false, -2, -2);
        DECLARE_CUSTOM_OP(cast, 1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(pad, 2, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(expose, -1, -1, true, 0, 0);
        DECLARE_CUSTOM_OP(Where, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(select, 3, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(shape_of, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(gather, 2, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(crelu, 1, 1, false, 0, 0);        
        DECLARE_CUSTOM_OP(crelu_bp, 2, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(biasadd_bp, 3, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(absoluteDifference, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(cosineDistance, 3, 1, false, 0, 2);
        DECLARE_CUSTOM_OP(squeeze, -1, -1, true, 0, 0);
        DECLARE_CUSTOM_OP(hingeLoss, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(huberLoss, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(logLoss, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(meanPairWsSqErr, 3, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(meanSqErr, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(sigmCrossEntropy, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(softmaxCrossEntropy, 3, 1, false, 1, 1);      
        DECLARE_CUSTOM_OP(batchnorm, 5, 1, false, 1, 2);
        DECLARE_CUSTOM_OP(unique, 1, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(lstmCell, 8, 2, false, 3, 2);
        DECLARE_CUSTOM_OP(set_seed, -2, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(get_seed, -2, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(shapes_of, -1, -1, false, 0, 0)
        DECLARE_CUSTOM_OP(sruCell, 4, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(gruCell, 5, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(diag, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(diag_part, 1, 1, false, 0, 0);

        // recurrent ops
        DECLARE_CUSTOM_OP(sru,         5, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(sru_logic,   5, 2, false, 0, 0);
        DECLARE_CUSTOM_OP(sru_bi,      5, 2, true,  0, 0);
        DECLARE_CUSTOM_OP(sru_bp,      8, 4, true,  0, 0);
        DECLARE_CUSTOM_OP(sru_bp_logic,8, 4, true,  0, 0);
        DECLARE_CUSTOM_OP(sru_bi_bp,   8, 4, true,  0, 0);
                
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(clipbynorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(clipbyavgnorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(cumsum, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(cumprod, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1);
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0);        
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0);        
        // DECLARE_CONFIGURABLE_OP(batchnorm_bp, 5, 1, true, 0, 1);                
        DECLARE_CONFIGURABLE_OP(conv3d_bp, 3, 1, false, 0, 7); // TODO: to be implemented        
        DECLARE_CONFIGURABLE_OP(ismax, 1, 1, false, 0, -1);
        DECLARE_CONFIGURABLE_OP(fill_as, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(reverse, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(axpy, 2, 1, false, -2, 0);
        DECLARE_CONFIGURABLE_OP(apply_sgd, 2, 1, true, -2, 0);
        DECLARE_CONFIGURABLE_OP(invert_permutation, 1, 1, false, 0, 0);        
        DECLARE_CONFIGURABLE_OP(matrix_set_diag, 2, 1, false, 0, 0)
        DECLARE_CONFIGURABLE_OP(betainc, 3, 1, false, 0, 0)

        // grad ops
        DECLARE_CONFIGURABLE_OP(sigmoid_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softsign_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(tanh_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softplus_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(relu_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(selu_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(lrelu_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(elu_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(cube_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(rectifiedtanh_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(rationaltanh_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(hardtanh_bp, 2, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(hardsigmoid_bp, 2, 1, true, 0, 0);

        DECLARE_CUSTOM_OP(firas_sparse, 1, 1, false, 0, -1);

        DECLARE_BOOLEAN_OP(lt_scalar, 2, true);
        DECLARE_BOOLEAN_OP(gt_scalar, 2, true);
        DECLARE_BOOLEAN_OP(lte_scalar, 2, true);
        DECLARE_BOOLEAN_OP(gte_scalar, 2, true);
        DECLARE_BOOLEAN_OP(eq_scalar, 2, true);
        DECLARE_BOOLEAN_OP(neq_scalar, 2, true);


        // list operations, basically all around NDArrayList
        DECLARE_LIST_OP(write_list, 2, 1, 0, -2);
        DECLARE_LIST_OP(stack_list, 1, 1, 0, 0);
        DECLARE_LIST_OP(read_list, 1, 1, 0, 1);
        DECLARE_LIST_OP(pick_list, 1, 1, -2, -2);
        DECLARE_LIST_OP(size_list, 1, 1, 0, 0);
        DECLARE_LIST_OP(create_list, 1, 2, 0, -2);
        DECLARE_LIST_OP(scatter_list, 1, 1, 0, -2);
        DECLARE_LIST_OP(split_list, 2, 1, 0, -2);
        DECLARE_LIST_OP(gather_list, 2, 1, 0, -2);
        DECLARE_LIST_OP(clone_list, 1, 1, 0, 0);


    }
}

#endif //LIBND4J_CUSTOMOPERATIONS_H
