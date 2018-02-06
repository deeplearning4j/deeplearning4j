//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/cross.h>

namespace nd4j {
namespace ops {

    CONFIGURABLE_OP_IMPL(log_poison_loss, 2, 1, true, 0, 0) {

        NDArray<T>* targets = INPUT_VARIABLE(0);
        NDArray<T>* input = INPUT_VARIABLE(1);
        bool computeFullLoss = false;

        if (block.numI() > 0)
            computeFullLoss = INT_ARG(0) != 0;
        
        REQUIRE_TRUE(targets->isSameShape(input), 0, "log_poison_loss: The shape of both input params should be equal.");

        NDArray<T>* output = OUTPUT_VARIABLE(0);
        if (!computeFullLoss)
            targets->template applyPairwiseTransform<simdOps::LogPoisonLoss<T>>(input, output, nullptr);
        else
            targets->template applyPairwiseTransform<simdOps::LogPoisonLossFull<T>>(input, output, nullptr);
/*
        std::unique_ptr<NDArray<T>> theFirst(input->dup('c'));
        // as mentioned in doc  res = exp(c) - z * c *[+ z * log(z) - z + 0.5 * log(2 * pi * z)]
        // the first we compute exp(c)
        input->template applyTransform<simdOps::Exp<T>>(theFirst.get()); // exp(c)
        //
        // then 0.5 * log(2 * pi * z)
        //
        std::unique_ptr<NDArray<T>> theLast(targets->dup());
        const T two_pi = (T)2.0 * (T)3.14159265358979323846;
        targets->template applyScalar<simdOps::Multiply<T>>(two_pi, theLast.get(), nullptr);
        std::unique_ptr<NDArray<T>> theNext(theLast->dup('c'));
       
        theLast->template applyTransform<simdOps::Log<T>>(theNext.get());
        std::unique_ptr<NDArray<T>> theTop(theNext->dup('c'));
        
        theNext->template applyScalar<simdOps::Multiply<T>>(T(0.5), theTop.get(), nullptr);

        // then z * log(z) - z
        std::unique_ptr<NDArray<T>> theMiddle(targets->dup('c'));
        targets->template applyTransform<simdOps::Log<T>>(theMiddle.get());
        targets->template applyPairwiseTransform<simdOps::Multiply<T>>(theMiddle.get(), theLast.get(), nullptr);
        theLast->template applyPairwiseTransform<simdOps::Subtract<T>>(targets, theNext.get(),  nullptr);
        // and add theMiddle and theLast
        theNext->template applyPairwiseTransform<simdOps::Add<T>>(theTop.get(), theLast.get(), nullptr);
        
        // then, z * c * theLast
        targets->template applyPairwiseTransform<simdOps::Multiply<T>>(input, theMiddle.get(), nullptr);
        theLast->template applyPairwiseTransform<simdOps::Multiply<T>>(theMiddle.get(), theTop.get(), nullptr);
        //targets->template applyPairwiseTransform<simdOps::Multiply<T>>(theTop.get(), theLast.get(), nullptr);
        // 
        //

        theFirst->template applyPairwiseTransform<simdOps::Subtract<T>>(theTop.get(), output, nullptr);
*/
        return ND4J_STATUS_OK;
    }
}
}