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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.interceptor.data.InterceptorPersistence;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public  class LayerBackpropGradientAdvice {
    @Advice.OnMethodEnter
    public static void enter( @Advice.Argument(0) INDArray epsilon) {
       if(epsilon != null) {
           InterceptorPersistence.addToBackwardPass(epsilon);
       }
    }

    @Advice.OnMethodExit
    public static void exit(@Advice.Return Pair<Gradient, INDArray> result) {
        if (result != null) {
            Gradient gradient = result.getFirst();
            if (gradient != null) {
                for (Map.Entry<String, INDArray> entry : gradient.gradientForVariable().entrySet()) {
                    INDArray gradientArray = entry.getValue();
                    if (gradientArray != null) {
                        InterceptorPersistence.addToBackwardPass(entry.getValue());
                    }
                }
            }

        }
    }
}
