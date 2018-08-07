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

package org.nd4j.linalg.dataset;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * Created by agibsonccc on 9/12/15.
 */
@RunWith(Parameterized.class)
public class StandardScalerTest extends BaseNd4jTest {
    public StandardScalerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Ignore
    @Test
    public void testScale() {
        StandardScaler scaler = new StandardScaler();
        DataSetIterator iter = new IrisDataSetIterator(10, 150);
        scaler.fit(iter);
        INDArray featureMatrix = new IrisDataSetIterator(150, 150).next().getFeatures();
        INDArray mean = featureMatrix.mean(0);
        INDArray std = featureMatrix.std(0);
        System.out.println(mean);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
