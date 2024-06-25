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

package org.nd4j.interceptor.advice;

import net.bytebuddy.asm.Advice;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.interceptor.data.InterceptorPersistence;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.nd4j.interceptor.data.InterceptorPersistence.finishCurrentForwardPass;

public class ComputationGraphForwardAdvice {
    public static final ThreadLocal<AtomicBoolean> calcForwardScope = ThreadLocal.withInitial(() -> new AtomicBoolean(false));

    public static boolean isCalcForwardScope() {
        return calcForwardScope.get().get();
    }

    @Advice.OnMethodEnter
    public static void enter(@Advice.Origin("#m") String methodName) {
        calcForwardScope.get().set(true);
    }

    @Advice.OnMethodExit
    public static void exit(@Advice.Origin("#m") String methodName) {
        calcForwardScope.get().set(false);
        finishCurrentForwardPass();
    }
}
