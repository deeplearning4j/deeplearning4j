package org.nd4j.interceptor.advice;

import net.bytebuddy.asm.Advice;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Pair;
import org.nd4j.interceptor.data.InterceptorPersistence;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ComputationGraphBackwardAdvice {
    public static final ThreadLocal<AtomicBoolean> calcBackpropScope = ThreadLocal.withInitial(() -> new AtomicBoolean(false));

    public static boolean isCalcBackpropScope() {
        return calcBackpropScope.get().get();
    }


    @Advice.OnMethodEnter
    public static void enter(@Advice.This Object thisObject,
                             @Advice.Origin("#m") String detailedOrigin) {
        calcBackpropScope.get().set(true);

    }

    @Advice.OnMethodExit
    public static void exit(@Advice.This Object thisObject,
                            @Advice.Origin("#m") String detailedOrigin) {
        InterceptorPersistence.finishCurrentBackwardPass();
        calcBackpropScope.get().set(false);

    }
}
