//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_CUSTOM_OP(hingeLoss, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(huberLoss, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(logLoss, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(meanPairWsSqErr, 3, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(meanSqErr, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(sigmCrossEntropy, 3, 1, false, 1, 1);
        DECLARE_CUSTOM_OP(softmaxCrossEntropy, 3, 1, false, 1, 1);  
        DECLARE_CUSTOM_OP(absoluteDifference, 3, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(cosineDistance, 3, 1, false, 0, 2);
    }
}