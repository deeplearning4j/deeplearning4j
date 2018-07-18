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

package org.nd4j.linalg.storage;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.compression.impl.Float16;
import org.nd4j.compression.impl.NoOp;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.storage.CompressedRamStorage;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CompressedRamStorageTests extends BaseNd4jTest {
    private CompressedRamStorage<Integer> halfsStorageInplace;
    private CompressedRamStorage<Integer> halfsStorageNIP;

    private CompressedRamStorage<Integer> noopStorageInplace;
    private CompressedRamStorage<Integer> noopStorageNIP;

    public CompressedRamStorageTests(Nd4jBackend backend) {
        super(backend);
    }


    @Before
    public void setUp() throws Exception {
        if (halfsStorageInplace == null) {
            halfsStorageInplace = new CompressedRamStorage.Builder<Integer>().setCompressor(new Float16())
                            .useInplaceCompression(true).build();

            halfsStorageNIP = new CompressedRamStorage.Builder<Integer>().setCompressor(new Float16())
                            .useInplaceCompression(false).build();
        }

        if (noopStorageInplace == null) {
            noopStorageInplace = new CompressedRamStorage.Builder<Integer>().setCompressor(new NoOp())
                            .useInplaceCompression(true).build();

            noopStorageNIP = new CompressedRamStorage.Builder<Integer>().setCompressor(new NoOp())
                            .useInplaceCompression(false).build();
        }
    }

    @Test
    public void testFP16StorageInplace1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        INDArray exp = array.dup();


        halfsStorageInplace.store(1, array);

        assertTrue(array.isCompressed());

        INDArray dec = halfsStorageInplace.get(1);

        assertEquals(exp, dec);
    }

    @Test
    public void testFP16StorageNIP1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        INDArray exp = array.dup();


        halfsStorageNIP.store(1, array);

        assertFalse(array.isCompressed());

        INDArray dec = halfsStorageNIP.get(1);

        assertEquals(exp, dec);
    }


    @Test
    public void testNoOpStorageInplace1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        INDArray exp = array.dup();

        noopStorageInplace.store(1, array);

        assertTrue(array.isCompressed());

        INDArray dec = noopStorageInplace.get(1);

        assertEquals(exp, dec);
    }

    @Test
    public void testNoOpStorageNIP1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        INDArray exp = array.dup();

        noopStorageNIP.store(1, array);

        assertFalse(array.isCompressed());

        INDArray dec = noopStorageNIP.get(1);

        assertEquals(exp, dec);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
