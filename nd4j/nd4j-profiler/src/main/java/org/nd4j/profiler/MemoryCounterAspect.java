package org.nd4j.profiler;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;

@Aspect
public class MemoryCounterAspect {


    @Around("execution(static * org.nd4j.nativeblas.OpaqueDataBuffer.allocateDataBuffer(..))")
    public Object allocateDataBuffer(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getSignature().getDeclaringTypeName();
        long size = (long) joinPoint.getArgs()[0];
        MemoryCounter.increment(className, size);
        return joinPoint.proceed();
    }



    @Around("execution(public static org.nd4j.nativeblas.OpaqueDataBuffer allocateDataBuffer(..))")
    public Object allocateMemory(ProceedingJoinPoint joinPoint) throws Throwable {
        String className = joinPoint.getSignature().getDeclaringTypeName();
        long size = (long) joinPoint.getArgs()[0];
        MemoryCounter.increment(className, size);
        return joinPoint.proceed();
    }


    @Around("execution(* allocateArray(..))")
    public Object allocateArrayMemory(ProceedingJoinPoint joinPoint) throws Throwable {
        System.out.println("Hello world");
        String className = joinPoint.getSignature().getDeclaringTypeName();
        long size = (long) joinPoint.getArgs()[0];
        MemoryCounter.increment(className, size);
        return joinPoint.proceed();
    }

    @Around("execution(* *deallocate(..))")
    public Object free(ProceedingJoinPoint joinPoint) throws Throwable {
        System.out.println("Hello world");
        String className = joinPoint.getSignature().getDeclaringTypeName();
        MemoryCounter.decrement(className, 1);

        return joinPoint.proceed();
    }


}
