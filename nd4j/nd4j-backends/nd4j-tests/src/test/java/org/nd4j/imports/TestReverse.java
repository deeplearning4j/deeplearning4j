/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.imports;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

public class TestReverse extends BaseNd4jTest {

    public TestReverse(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testReverse(){

        INDArray in = Nd4j.createFromArray(new double[]{1,2,3,4,5,6});
        INDArray out = Nd4j.create(DataType.DOUBLE, 6);

        DynamicCustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(in)
                .addOutputs(out)
                .addIntegerArguments(0)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);
    }

    @Test
    public void testReverse2(){

        INDArray in = Nd4j.createFromArray(new double[]{1,2,3,4,5,6});
        INDArray axis = Nd4j.scalar(0);
        INDArray out = Nd4j.create(DataType.DOUBLE, 6);

        DynamicCustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(in, axis)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);
    }
}
