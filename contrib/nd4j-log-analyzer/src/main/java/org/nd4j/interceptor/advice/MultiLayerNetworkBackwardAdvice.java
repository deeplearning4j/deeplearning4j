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

public class MultiLayerNetworkBackwardAdvice {


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
