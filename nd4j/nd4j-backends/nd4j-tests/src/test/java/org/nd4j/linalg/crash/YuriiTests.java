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

package org.nd4j.linalg.crash;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.LegacyPooling2D;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

@Slf4j
@RunWith(Parameterized.class)
public class YuriiTests extends BaseNd4jTest {
    public YuriiTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void legacyPooling2dTest_double(){

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(DataType.DOUBLE, new int[]{1,1,3,3});
        INDArray out = Nd4j.create(DataType.DOUBLE, 1,1,2,2).assign(-119);

        Nd4j.getExecutioner().commit();

        val op = new LegacyPooling2D(in, 2, 2, 1, 1, 0, 0, 1, 1, true, LegacyPooling2D.Pooling2DType.MAX, 0.0, out);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();
        System.out.println(in);
        System.out.println(out);
    }


    @Test
    public void legacyPooling2dTes_float(){

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(DataType.FLOAT, new int[]{1,1,3,3});
        INDArray out = Nd4j.create(DataType.FLOAT, 1,1,2,2);

        val op = new LegacyPooling2D(in, 2, 2, 1, 1, 0, 0, 1, 1, true, LegacyPooling2D.Pooling2DType.MAX, 0.0, out);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();
        System.out.println(in);
        System.out.println(out);
    }

    @Test
    public void legacyPooling2dTes_half(){

        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(DataType.HALF, new int[]{1,1,3,3});
        INDArray out = Nd4j.create(DataType.HALF, 1,1,2,2);

        val op = new LegacyPooling2D(in, 2, 2, 1, 1, 0, 0, 1, 1, true, LegacyPooling2D.Pooling2DType.MAX, 0.0, out);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();
        System.out.println(in);
        System.out.println(out);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
