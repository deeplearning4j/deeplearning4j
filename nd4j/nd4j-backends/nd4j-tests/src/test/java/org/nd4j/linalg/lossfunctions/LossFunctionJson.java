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

package org.nd4j.linalg.lossfunctions;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 09/09/2016.
 */
public class LossFunctionJson extends BaseNd4jTest {

    public LossFunctionJson(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testJsonSerialization() throws Exception {

        INDArray w = Nd4j.create(new double[] {1.0, 2.0, 3.0});

        ILossFunction[] lossFns = new ILossFunction[] {new LossBinaryXENT(), new LossBinaryXENT(w),
                        new LossCosineProximity(), new LossHinge(), new LossKLD(), new LossL1(), new LossL1(w),
                        new LossL2(), new LossL2(w), new LossMAE(), new LossMAE(w), new LossMAPE(), new LossMAPE(w),
                        new LossMCXENT(), new LossMCXENT(w), new LossMSE(), new LossMSE(w), new LossMSLE(),
                        new LossMSLE(w), new LossNegativeLogLikelihood(), new LossNegativeLogLikelihood(w),
                        new LossPoisson(), new LossSquaredHinge(), new LossMultiLabel()};

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        for (ILossFunction lf : lossFns) {
            String asJson = mapper.writeValueAsString(lf);
            //            System.out.println(asJson);

            ILossFunction fromJson = mapper.readValue(asJson, ILossFunction.class);
            assertEquals(lf, fromJson);
        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
