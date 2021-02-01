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

package org.nd4j.autodiff.opvalidation;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class ActivationGradChecks extends BaseOpValidation {

    public ActivationGradChecks(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testActivationGradientCheck1(){
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("x", Nd4j.rand(DataType.DOUBLE, 3, 4));
        SDVariable tanh = sd.math().tanh("tanh", in);
        SDVariable loss = tanh.std(true);

        GradCheckUtil.ActGradConfig c = GradCheckUtil.ActGradConfig.builder()
                .sd(sd)
                .activationGradsToCheck(Collections.singletonList("tanh"))
                .build();

        boolean ok = GradCheckUtil.checkActivationGradients(c);

        assertTrue(ok);
    }

    @Test
    public void testActivationGradientCheck2(){
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.DOUBLE, 3, 4);
        SDVariable y = sd.var("y", Nd4j.rand(DataType.DOUBLE, 4, 5));
        SDVariable mmul = x.mmul("mmul", y);
        SDVariable sigmoid = sd.math().tanh("sigmoid", mmul);
        SDVariable loss = sigmoid.std(true);

        Map<String, INDArray> m = new HashMap<>();
        m.put("x", Nd4j.rand(DataType.DOUBLE, 3, 4));

        GradCheckUtil.ActGradConfig c = GradCheckUtil.ActGradConfig.builder()
                .sd(sd)
                .placeholderValues(m)
                .activationGradsToCheck(Arrays.asList("sigmoid", "mmul"))
                .subset(GradCheckUtil.Subset.RANDOM)
                .maxPerParam(10)
                .build();

        boolean ok = GradCheckUtil.checkActivationGradients(c);

        assertTrue(ok);
    }
}
