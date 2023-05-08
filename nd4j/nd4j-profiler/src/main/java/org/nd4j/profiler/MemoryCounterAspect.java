/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.profiler;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.bytedeco.javacpp.Pointer;

/**
 * Aspect for tracking memory allocation and deallocation
 * as well as current memory usage at a given
 * time when System.gc() is called.
 *
 * @author Adam Gibson
 */
@Aspect
public class MemoryCounterAspect {

    /**
     * Track memory allocation for pointers.
     * @param joinPoint when a  new pointer is created
     * @return the result of proceeding with the execution
     * of the passed in join point
     * @throws Throwable
     */
    @Around("execution(org.bytedeco..*.new(..))")
    public Object allocateMemory(ProceedingJoinPoint joinPoint) throws Throwable {
        if (joinPoint != null && joinPoint.getSignature() != null && joinPoint.getTarget() instanceof Pointer) {
            String className = joinPoint.getSignature().getDeclaringTypeName();
            long currMemory = Pointer.physicalBytes();
            Object ret = joinPoint.proceed();
            long after = Pointer.physicalBytes();
            MemoryCounter.increment(className, after - currMemory);
            return ret;
        }

        return joinPoint.proceed();
    }

    /**
     * Track memory  deallocation
     * for pointers.
     * @param joinPoint when a pointer is deallocated
     * @return
     * @throws Throwable
     */
    @Around("execution(* org.bytedeco..*.*deallocate*(..))")
    public Object deallocate(ProceedingJoinPoint joinPoint) throws Throwable {
        if (joinPoint != null && joinPoint.getSignature() != null && joinPoint.getTarget() instanceof Pointer) {
            String className = joinPoint.getSignature().getDeclaringTypeName();
            long currMemory = Pointer.physicalBytes();
            Object ret = joinPoint.proceed();
            long after = Pointer.physicalBytes();
            MemoryCounter.decrement(className, currMemory - after);
            return ret;
        }

        return joinPoint.proceed();
    }

    /**
     * Track memory usage when System.gc() is called.
     * @param joinPoint when System.gc() is called
     * @return
     * @throws Throwable
     */
    @Around("execution(* java.lang.System.gc(..))")
    public Object trackGcActivity(ProceedingJoinPoint joinPoint) throws Throwable {
        long beforeGcMemory = Pointer.physicalBytes();
        Object ret = joinPoint.proceed();
        long afterGcMemory = Pointer.physicalBytes();
        MemoryCounter.recordGC(Math.abs(beforeGcMemory - afterGcMemory));
        return ret;
    }
}
