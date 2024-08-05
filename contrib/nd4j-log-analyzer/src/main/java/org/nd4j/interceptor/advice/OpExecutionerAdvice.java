package org.nd4j.interceptor.advice;

import net.bytebuddy.asm.Advice;
import org.nd4j.interceptor.util.InterceptorUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;

public class OpExecutionerAdvice {
    @Advice.OnMethodExit
    public static void exit(@Advice.AllArguments Object[] args) {
        if (args != null && args.length > 0) {
            Object opOrCustomOp = args[0];
            if (opOrCustomOp instanceof Op) {
                Op op = (Op) opOrCustomOp;
                InterceptorUtils.logOpExecution(op);
            }
        }
    }

    public static void error(@Advice.Thrown Throwable t) {
        t.printStackTrace();
    }
}
