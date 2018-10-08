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

package org.deeplearning4j.clustering.randomprojection;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class RPUtilsTest {

    @Test
    public void testDistanceComputeBatch() {
        INDArray x = Nd4j.linspace(1,4,4);
        INDArray y = Nd4j.linspace(1,16,16).reshape(4,4);
        INDArray result = Nd4j.create(4);
        INDArray distances = RPUtils.computeDistanceMulti("euclidean",x,y,result);
        INDArray scalarResult = Nd4j.scalar(1.0);
        for(int i = 0; i < result.length(); i++) {
            double dist = RPUtils.computeDistance("euclidean",x,y.slice(i),scalarResult);
            assertEquals(dist,distances.getDouble(i),1e-3);
        }
    }

}
