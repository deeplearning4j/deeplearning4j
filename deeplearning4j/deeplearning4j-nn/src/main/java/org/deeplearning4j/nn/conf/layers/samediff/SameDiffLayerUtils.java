/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.layers.samediff;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

import java.util.HashMap;
import java.util.Map;

public class SameDiffLayerUtils {

    private static Map<Class<?>, Activation> activationMap;

    private SameDiffLayerUtils(){ }

    public static Activation fromIActivation(IActivation a){

        if(activationMap == null){
            Map<Class<?>,Activation> m = new HashMap<>();
            for(Activation act : Activation.values()){
                m.put(act.getActivationFunction().getClass(), act);
            }
            activationMap = m;
        }

        return activationMap.get(a.getClass());
    }

}
