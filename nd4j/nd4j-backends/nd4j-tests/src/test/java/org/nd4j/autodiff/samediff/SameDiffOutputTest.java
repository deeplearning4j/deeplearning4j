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

package org.nd4j.autodiff.samediff;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Sgd;

import static org.junit.Assert.assertTrue;

public class SameDiffOutputTest extends BaseNd4jTest {

    public SameDiffOutputTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void outputTest(){
        DataSet data = new DataSet(Nd4j.zeros(10, 10), Nd4j.zeros(10, 10));
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, 10, 10);
        SDVariable out = in.add("out", 2);

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Sgd(3e-1))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping()
                .build();

        sd.setTrainingConfig(conf);

        INDArray output = sd.output(data, "out").get("out");

        assertTrue("output != input + 2", output.equalsWithEps(
                Nd4j.zeros(10, 10).add(2).castTo(DataType.FLOAT),
                0.0001));
    }

    @Override
    public char ordering() {
        return 'c';
    }

}
