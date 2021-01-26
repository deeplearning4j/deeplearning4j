/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.List;

import static org.junit.Assert.assertNotNull;

public class KerasTestUtils {

    private KerasTestUtils(){ }

    public static double getL1(BaseLayer layer) {
        List<Regularization> l = layer.getRegularization();
        return getL1(l);
    }

    public static double getL1(List<Regularization> l){
        L1Regularization l1Reg = null;
        for(Regularization reg : l){
            if(reg instanceof L1Regularization)
                l1Reg = (L1Regularization) reg;
        }
        assertNotNull(l1Reg);
        return l1Reg.getL1().valueAt(0,0);
    }

    public static double getL2(BaseLayer layer) {
        List<Regularization> l = layer.getRegularization();
        return getL2(l);
    }

    public static double getL2(List<Regularization> l){
        L2Regularization l2Reg = null;
        for(Regularization reg : l){
            if(reg instanceof L2Regularization)
                l2Reg = (L2Regularization) reg;
        }
        assertNotNull(l2Reg);
        return l2Reg.getL2().valueAt(0,0);
    }

    public static double getL1(AbstractSameDiffLayer layer){
        return getL1(layer.getRegularization());
    }

    public static double getL2(AbstractSameDiffLayer layer){
        return getL2(layer.getRegularization());
    }

}
