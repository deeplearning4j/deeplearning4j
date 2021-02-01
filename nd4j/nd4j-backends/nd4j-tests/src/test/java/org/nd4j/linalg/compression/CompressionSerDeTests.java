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

package org.nd4j.linalg.compression;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.ByteArrayInputStream;

import static org.junit.Assert.assertEquals;

/**
 * Tests for SerDe on compressed arrays
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressionSerDeTests extends BaseNd4jTest {
    public CompressionSerDeTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testAutoDecompression2() throws Exception {
        INDArray array = Nd4j.linspace(1, 10, 11, DataType.DOUBLE);

        INDArray compressed = Nd4j.getCompressor().compress(array, "GZIP");

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Nd4j.write(bos, compressed);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());

        System.out.println("Restoring -------------------------");

        INDArray result = Nd4j.read(bis);

        System.out.println("Decomp -------------------------");

        INDArray decomp = Nd4j.getCompressor().decompress(result);

        assertEquals(array, decomp);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
