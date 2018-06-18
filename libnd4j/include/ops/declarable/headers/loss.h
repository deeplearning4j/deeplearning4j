//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_LOSS_H
#define LIBND4J_HEADERS_LOSS_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
namespace ops {
    
    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of hinge loss function max(0, 1 - labels*logits)
       * 
       * Input arrays: 
       *    0: logits - logits, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels and must be broadcastable to labels.
       *    2: labels - ground truth vales, expected to be 0. or 1., type float.
       *       Must have the same shape as logits.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as logits.
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as logits or just single scalar, depending on reduction mode (see input integer argument)
       */               
        #if NOT_EXCLUDED(OP_hinge_loss)
        DECLARE_CUSTOM_OP(hinge_loss, 3, 1, false, 0, 1);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of Huber loss function:
       *    0.5 * (labels-predictions)^2                                if |labels-predictions| <= delta
       *    0.5 * delta^2 + delta * (|labels-predictions| - delta)      if |labels-predictions| >  delta
       * 
       * Input arrays: 
       *    0: predictions - the predicted values, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels, and must be broadcastable to labels.
       *    2: labels - ground truth vales, type float.
       *       Must have the same shape as predictions.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as predictions
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       *  Input float arguments:
       *    0: point where the huber loss function changes from a quadratic to linear.
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as predictions or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_huber_loss)
        DECLARE_CUSTOM_OP(huber_loss, 3, 1, false, 1, 1);
        #endif

    
    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of logarithmic loss function ( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
       * 
       * Input arrays: 
       *    0: predictions - the predicted values, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels, and must be broadcastable to labels.
       *    2: labels - ground truth vales, type float.
       *       Must have the same shape as predictions.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as predictions
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       *  Input float arguments:
       *    0: a small increment to add to avoid taking a log of zero. 
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as predictions or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_log_loss)
        DECLARE_CUSTOM_OP(log_loss, 3, 1, false, 1, 1);
        #endif
    

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of pairwise-errors-squared loss function 
       * 
       * Input arrays: 
       *    0: predictions - the predicted values, type float.
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels and must be broadcastable to labels.
       *    2: labels - ground truth vales, type float.
       *       Must have the same shape as predictions.    
       *  
       * Output array: 
       *    0: loss value, it is just single scalar, type float.
       */     
        #if NOT_EXCLUDED(OP_mean_pairwssqerr_loss)
        DECLARE_CUSTOM_OP(mean_pairwssqerr_loss, 3, 1, false, 0, 0);
        #endif

    
    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of Sum-of-Squares loss function 1/N * sum_{i}^{N}(predictions_i - labels_i)^2
       * 
       * Input arrays: 
       *    0: predictions - the predicted values, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels and must be broadcastable to labels.
       *    2: labels - ground truth vales, type float.
       *       Must have the same shape as predictions.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as predictions
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as predictions or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_mean_sqerr_loss)
        DECLARE_CUSTOM_OP(mean_sqerr_loss, 3, 1, false, 0, 1);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of sigmoid cross-entropy loss function max(logits, 0.) - logits * labels + log(1. + exp(-abs(logits))); 
       * 
       * Input arrays: 
       *    0: logits - logits, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels, and must be broadcastable to labels.
       *    2: labels - ground truth vales, expected to be 0. or 1., type float.
       *       Must have the same shape as logits.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as logits.
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       *  Input float arguments:
       *    0: smoothing value, if it is greater than 0 then apply smoothing to the labels (smooth the labels towards 1/2): new_labels = labels * (1 - labelsSmoothing)+ 0.5 * labelsSmoothing
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as logits or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_sigm_cross_entropy_loss)
        DECLARE_CUSTOM_OP(sigm_cross_entropy_loss, 3, 1, false, 1, 1);
        #endif
    

    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of softmax cross-entropy loss function max(logits, 0.) - logits * labels + log(1. + exp(-abs(logits))); 
       * 
       * Input arrays: 
       *    0: logits - logits, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels, and must be broadcastable to labels.
       *    2: labels - ground truth vales, expected to be 0. or 1., type float.
       *       Must have the same shape as logits.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as logits.
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       *  Input float arguments:
       *    0: smoothing value, if it is greater than 0 then apply smoothing to the labels (smooth the labels towards 1/numClasses):  new_labels = labels * (1 - labelsSmoothing) + labelsSmoothing / numClasses
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with shape as in logits except last dimension is equal to unity or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_softmax_cross_entropy_loss)
        DECLARE_CUSTOM_OP(softmax_cross_entropy_loss, 3, 1, false, 1, 1);  
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of Absolute Difference loss function |predictions - labels|
       * 
       * Input arrays: 
       *    0: predictions - the predicted values, type float.
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels and must be broadcastable to labels.
       *    2: labels - ground truth vales, type float.
       *       Must have the same shape as predictions.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as predictions
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as predictions or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_absolute_difference_loss)
        DECLARE_CUSTOM_OP(absolute_difference_loss, 3, 1, false, 0, 1);
        #endif


    //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of cosine-distance loss function 1. - (predictions * labels).reduce_sum_along(dimension)
       * 
       * Input arrays: 
       *    0: predictions - the predicted values, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels and must be broadcastable to labels.
       *    2: labels - ground truth vales, type float.
       *       Must have the same shape as predictions.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as predictions
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *    1: dimension along which the cosine distance is computed.
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with the same shape as predictions or just single scalar, depending on reduction mode (see input integer argument)
       */         
        #if NOT_EXCLUDED(OP_cosine_distance_loss)
        DECLARE_CUSTOM_OP(cosine_distance_loss, 3, 1, false, 0, 2);
        #endif

          //////////////////////////////////////////////////////////////////////////
    /**
       * Implementation of softmax cross-entropy loss function max(logits, 0.) - logits * labels + log(1. + exp(-abs(logits))); 
       * 
       * Input arrays: 
       *    0: logits - logits, type float
       *    1: weights - is used for weighting (multiplying) of loss values, type float. 
       *       Can be single scalar or has the same rank as labels, and must be broadcastable to labels.
       *    2: labels - ground truth vales, expected to be 0. or 1., type float.
       *       Must have the same shape as logits.    
       *  
       *  Input integer arguments:
       *    0: type of reduction to apply to loss
       *       0 - "none", unreduced weighted losses with the same shape as logits.
       *       1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
       *       2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
       *       3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
       *
       *  Input float arguments:
       *    0: smoothing value, if it is greater than 0 then apply smoothing to the labels (smooth the labels towards 1/numClasses):  new_labels = labels * (1 - labelsSmoothing) + labelsSmoothing / numClasses
       *
       * Output array: 
       *    0: loss values, type float.
       *       Can be an array with shape as in logits except last dimension is equal to unity or just single scalar, depending on reduction mode (see input integer argument)
       */      
        #if NOT_EXCLUDED(OP_softmax_cross_entropy_loss_with_logits)
        DECLARE_CUSTOM_OP(softmax_cross_entropy_loss_with_logits, 3, 1, false, 1, 1);  
        #endif

}
}

#endif