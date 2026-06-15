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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.eclipse.deeplearning4j.tests.extensions.BackendTest;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertNull;

@BackendTest(description = "Isolation coverage for dynamic_partition backprop output wiring")
public class DynamicPartitionGradientIsolationTest {

    @Test
    public void testDynamicPartitionGradientCheckDoesNotCrash() {
        DataType previousDefault = Nd4j.defaultFloatingPointType();
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        try {
            SameDiff sd = SameDiff.create();

            INDArray input = Nd4j.create(new double[]{4, 3, 5, 7, 8, 0});
            INDArray partitions = Nd4j.create(new double[]{1, 0, 1, 0, 0, 1}).castTo(DataType.INT);
            int numPartitions = 2;

            SDVariable in = sd.var("in", DataType.DOUBLE, 6);
            SDVariable sdPartitions = sd.var("partitions", DataType.INT, 6);

            INDArray expOut0 = Nd4j.create(DataType.DOUBLE, 3);
            INDArray expOut1 = Nd4j.create(DataType.DOUBLE, 3);
            Nd4j.exec(DynamicCustomOp.builder("dynamic_partition")
                    .addInputs(input, partitions)
                    .addIntegerArguments(numPartitions)
                    .addOutputs(expOut0, expOut1)
                    .build());

            SDVariable[] outputs = sd.dynamicPartition(new String[]{"dp0", "dp1"}, in, sdPartitions, numPartitions);
            SDVariable merged = sd.math().mergeAdd(outputs);
            sd.mean("loss", merged);

            sd.associateArrayWithVariable(input, in);
            sd.associateArrayWithVariable(partitions, sdPartitions);

            String err = OpValidation.validate(new TestCase(sd)
                    .expectedOutput("dp0", expOut0)
                    .expectedOutput("dp1", expOut1)
                    .gradCheckSkipVariables("partitions")
                    .gradientCheck(true));
            assertNull(err, err);
        } finally {
            Nd4j.setDefaultDataTypes(previousDefault, previousDefault);
        }
    }
}
