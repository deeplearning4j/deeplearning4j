/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.compression;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class CompressionTests extends BaseNd4jTest {

    public CompressionTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testCompressionDescriptorSerde() {
        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(4);
        descriptor.setOriginalElementSize(4);
        descriptor.setNumberOfElements(4);
        descriptor.setCompressionAlgorithm("GZIP");
        descriptor.setOriginalLength(4);
        descriptor.setOriginalDataType(DataType.LONG);
        descriptor.setCompressionType(CompressionType.LOSSY);
        ByteBuffer toByteBuffer = descriptor.toByteBuffer();
        CompressionDescriptor fromByteBuffer = CompressionDescriptor.fromByteBuffer(toByteBuffer);
        assertEquals(descriptor, fromByteBuffer);
    }

    @Test
    public void testGzipInPlaceCompression() {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        Nd4j.getCompressor().setDefaultCompression("GZIP");
        Nd4j.getCompressor().compressi(array);
        assertTrue(array.isCompressed());
        Nd4j.getCompressor().decompressi(array);
        assertFalse(array.isCompressed());
    }

    @Test
    public void testGzipCompression1() {
        INDArray array = Nd4j.linspace(1, 10000, 20000, DataType.FLOAT);
        INDArray exp = array.dup();

        BasicNDArrayCompressor.getInstance().setDefaultCompression("GZIP");

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataType.COMPRESSED, compr.data().dataType());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(exp, array);
        assertEquals(exp, decomp);
    }

    @Test
    public void testNoOpCompression1() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        INDArray array = Nd4j.linspace(1, 10000, 20000, DataType.FLOAT);
        INDArray exp = Nd4j.linspace(1, 10000, 20000, DataType.FLOAT);
        INDArray mps = Nd4j.linspace(1, 10000, 20000, DataType.FLOAT);

        BasicNDArrayCompressor.getInstance().setDefaultCompression("NOOP");

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataType.COMPRESSED, compr.data().dataType());
        assertTrue(compr.isCompressed());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(DataType.FLOAT, decomp.data().dataType());
        assertFalse(decomp.isCompressed());
        assertFalse(decomp.data() instanceof CompressedDataBuffer);
        assertFalse(exp.data() instanceof CompressedDataBuffer);
        assertFalse(exp.isCompressed());
        assertFalse(array.data() instanceof CompressedDataBuffer);

        assertEquals(exp, decomp);
    }

    @Test
    public void testJVMCompression3() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        INDArray exp = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f}).reshape(1,-1);

        BasicNDArrayCompressor.getInstance().setDefaultCompression("NOOP");

        INDArray compressed = BasicNDArrayCompressor.getInstance().compress(new float[] {1f, 2f, 3f, 4f, 5f});
        assertNotEquals(null, compressed.data());
        assertNotEquals(null, compressed.shapeInfoDataBuffer());
        assertTrue(compressed.isCompressed());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compressed);

        assertEquals(exp, decomp);
    }


    @Test
    public void testThresholdCompressionZ() {
        INDArray initial = Nd4j.create(1, 16384);
        for (int i = 0; i < 96; i++)
            initial.putScalar(i * 20, 1.0f);


        INDArray exp = Nd4j.create(1, 16384);
        for (int i = 0; i < 96; i++)
            exp.putScalar(i * 20, 0.1f);

        INDArray exp_d = Nd4j.create(1, 16384);
        for (int i = 0; i < 96; i++)
            exp_d.putScalar(i * 20, 0.9f);

        NDArrayCompressor compressor = Nd4j.getCompressor().getCompressor("THRESHOLD");
        compressor.configure(0.9);

        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 0.9);

        assertEquals(exp, initial);

        log.info("Compressed length: {}", compressed.data().length());
        //        log.info("Compressed: {}", Arrays.toString(compressed.data().asInt()));

        INDArray decompressed = Nd4j.create(1, initial.length());
        Nd4j.getExecutioner().thresholdDecode(compressed, decompressed);

        log.info("Decompressed length: {}", decompressed.lengthLong());

        assertEquals(exp_d, decompressed);
    }


    @Ignore
    @Test
    public void testThresholdCompression0() {
        INDArray initial = Nd4j.rand(new int[] {1, 150000000}, 119L);

        log.info("DTYPE: {}", Nd4j.dataType());

        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(2 * 1024L * 1024L * 1024L)
                        .overallocationLimit(0).policyAllocation(AllocationPolicy.STRICT)
                        .policyLearning(LearningPolicy.NONE).policyReset(ResetPolicy.BLOCK_LEFT).build();


        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "IIIA")) {
            INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial.dup(), 0.999);
        }

        long timeS = 0;
        for (int i = 0; i < 100; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "IIIA")) {
                INDArray d = initial.dup();
                long time1 = System.nanoTime();
                INDArray compressed = Nd4j.getExecutioner().thresholdEncode(d, 0.999);
                long time2 = System.nanoTime();
                timeS += (time2 - time1) / 1000;
            }
        }


        log.info("Elapsed time: {} us", (timeS) / 100);
    }

    @Test
    @Ignore
    public void testThresholdCompression1() {
        INDArray initial = Nd4j.create(new float[] {0.0f, 0.0f, 1e-3f, -1e-3f, 0.0f, 0.0f});
        INDArray exp_0 = Nd4j.create(DataType.FLOAT, 6);
        INDArray exp_1 = initial.dup();

        NDArrayCompressor compressor = Nd4j.getCompressor().getCompressor("THRESHOLD");
        compressor.configure(1e-3);

        INDArray compressed = compressor.compress(initial);

        log.info("Initial array: {}", Arrays.toString(initial.data().asFloat()));

        INDArray decompressed = compressor.decompress(compressed);

        assertEquals(exp_1, decompressed);
        assertEquals(exp_0, initial);
    }

    @Test
    public void testThresholdCompression2() {
        INDArray initial = Nd4j.create(new double[] {1.0, 2.0, 0.0, 0.0, -1.0, -1.0});
        INDArray exp_0 = Nd4j.create(new double[] {1.0 - 1e-3, 2.0 - 1e-3, 0.0, 0.0, -1.0 + 1e-3, -1.0 + 1e-3});
        INDArray exp_1 = Nd4j.create(new double[] {1e-3, 1e-3, 0.0, 0.0, -1e-3, -1e-3});

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        //NDArray compressed = Nd4j.getCompressor().compress(initial, "THRESHOLD");
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1e-3f);

        log.info("Initial array: {}", Arrays.toString(initial.data().asFloat()));

        assertEquals(exp_0, initial);

        INDArray decompressed = Nd4j.create(DataType.DOUBLE, initial.length());
        Nd4j.getExecutioner().thresholdDecode(compressed, decompressed);

        log.info("Decompressed array: {}", Arrays.toString(decompressed.data().asFloat()));

        assertEquals(exp_1, decompressed);
    }

    @Test
    public void testThresholdCompression3() {
        INDArray initial = Nd4j.create(new double[] {-1.0, -2.0, 0.0, 0.0, 1.0, 1.0});
        INDArray exp_0 = Nd4j.create(new double[] {-1.0 + 1e-3, -2.0 + 1e-3, 0.0, 0.0, 1.0 - 1e-3, 1.0 - 1e-3});
        INDArray exp_1 = Nd4j.create(new double[] {-1e-3, -1e-3, 0.0, 0.0, 1e-3, 1e-3});

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1e-3f);

        INDArray copy = compressed.unsafeDuplication();

        log.info("Initial array: {}", Arrays.toString(initial.data().asFloat()));

        assertEquals(exp_0, initial);

        INDArray decompressed = Nd4j.create(DataType.DOUBLE, initial.length());
        Nd4j.getExecutioner().thresholdDecode(compressed, decompressed);

        log.info("Decompressed array: {}", Arrays.toString(decompressed.data().asFloat()));

        assertEquals(exp_1, decompressed);

        INDArray decompressed_copy = Nd4j.create(DataType.DOUBLE, initial.length());
        Nd4j.getExecutioner().thresholdDecode(copy, decompressed_copy);

        assertFalse(decompressed == decompressed_copy);
        assertEquals(decompressed, decompressed_copy);
    }

    @Test
    public void testThresholdCompression4() {
        INDArray initial = Nd4j.create(new double[] {1e-4, -1e-4, 0.0, 0.0, 1e-4, -1e-4});
        INDArray exp_0 = initial.dup();


        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1e-3f);


        log.info("Initial array: {}", Arrays.toString(initial.data().asFloat()));

        assertEquals(exp_0, initial);

        assertNull(compressed);
    }


    @Test
    public void testThresholdCompression5() {
        INDArray initial = Nd4j.ones(1000);
        INDArray exp_0 = initial.dup();

        Nd4j.getExecutioner().commit();

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1.0f, 100);

        assertEquals(104, compressed.data().length());

        assertNotEquals(exp_0, initial);

        assertEquals(900, initial.sumNumber().doubleValue(), 0.01);
    }

    @Test
    public void testThresholdCompression6() {
        INDArray initial = Nd4j.create(new double[] {1.0, 2.0, 0.0, 0.0, -1.0, -1.0});
        INDArray exp_0 = Nd4j.create(new double[] {1.0 - 1e-3, 2.0 - 1e-3, 0.0, 0.0, -1.0 + 1e-3, -1.0 + 1e-3});
        INDArray exp_1 = Nd4j.create(new double[] {1e-3, 1e-3, 0.0, 0.0, -1e-3, -1e-3});
        INDArray exp_2 = Nd4j.create(new double[] {2e-3, 2e-3, 0.0, 0.0, -2e-3, -2e-3});

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        //NDArray compressed = Nd4j.getCompressor().compress(initial, "THRESHOLD");
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1e-3f);

        log.info("Initial array: {}", Arrays.toString(initial.data().asFloat()));

        assertEquals(exp_0, initial);

        INDArray decompressed = Nd4j.create(DataType.DOUBLE, initial.length());
        Nd4j.getExecutioner().thresholdDecode(compressed, decompressed);

        log.info("Decompressed array: {}", Arrays.toString(decompressed.data().asFloat()));

        assertEquals(exp_1, decompressed);

        Nd4j.getExecutioner().thresholdDecode(compressed, decompressed);

        assertEquals(exp_2, decompressed);
    }



    @Test
    public void testThresholdSerialization1() throws Exception {
        INDArray initial = Nd4j.create(new double[] {-1.0, -2.0, 0.0, 0.0, 1.0, 1.0});
        INDArray exp_0 = Nd4j.create(new double[] {-1.0 + 1e-3, -2.0 + 1e-3, 0.0, 0.0, 1.0 - 1e-3, 1.0 - 1e-3});
        INDArray exp_1 = Nd4j.create(new double[] {-1e-3, -1e-3, 0.0, 0.0, 1e-3, 1e-3});

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1e-3f);

        assertEquals(exp_0, initial);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Nd4j.write(baos, compressed);

        INDArray serialized = Nd4j.read(new ByteArrayInputStream(baos.toByteArray()));

        INDArray decompressed_copy = Nd4j.create(DataType.DOUBLE, initial.length());
        Nd4j.getExecutioner().thresholdDecode(serialized, decompressed_copy);

        assertEquals(exp_1, decompressed_copy);
    }

    @Test
    public void testBitmapEncoding1() {
        INDArray initial = Nd4j.create(new float[] {0.0f, 0.0f, 1e-3f, -1e-3f, 0.0f, 0.0f});
        INDArray exp_0 = Nd4j.create(DataType.FLOAT, 6);
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);

        log.info("Encoded: {}", Arrays.toString(enc.data().asInt()));

        assertEquals(exp_0, initial);
        assertEquals(5, enc.data().length());

        log.info("Encoded: {}", Arrays.toString(enc.data().asInt()));

        INDArray target = Nd4j.create(DataType.FLOAT, 6);
        Nd4j.getExecutioner().bitmapDecode(enc, target);

        log.info("Target: {}", Arrays.toString(target.data().asFloat()));
        assertEquals(exp_1, target);
    }

    @Test
    public void testBitmapEncoding1_1() {
        INDArray initial = Nd4j.create(15);
        INDArray exp_0 = Nd4j.create(6);
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);

        //assertEquals(exp_0, initial);
        assertEquals(5, enc.data().length());

        initial = Nd4j.create(31);

        enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);

        assertEquals(6, enc.data().length());

        initial = Nd4j.create(32);

        enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);

        assertEquals(7, enc.data().length());
    }

    @Test
    public void testBitmapEncoding2() {
        INDArray initial = Nd4j.create(40000000);
        INDArray target = Nd4j.create(initial.length());

        initial.addi(1e-3);

        long time1 = System.currentTimeMillis();
        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);
        long time2 = System.currentTimeMillis();


        Nd4j.getExecutioner().bitmapDecode(enc, target);
        long time3 = System.currentTimeMillis();

        log.info("Encode time: {}", time2 - time1);
        log.info("Decode time: {}", time3 - time2);
    }


    @Test
    public void testBitmapEncoding3() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        INDArray initial = Nd4j.create(new float[] {0.0f, -6e-4f, 1e-3f, -1e-3f, 0.0f, 0.0f});
        INDArray exp_0 = Nd4j.create(new float[] {0.0f, -1e-4f, 0.0f, 0.0f, 0.0f, 0.0f});
        INDArray exp_1 = Nd4j.create(new float[] {0.0f, -5e-4f, 1e-3f, -1e-3f, 0.0f, 0.0f});

        DataBuffer ib = Nd4j.getDataBufferFactory().createInt(5);
        INDArray enc = Nd4j.createArrayFromShapeBuffer(ib, initial.shapeInfoDataBuffer());

        long elements = Nd4j.getExecutioner().bitmapEncode(initial, enc, 1e-3);
        log.info("Encoded: {}", Arrays.toString(enc.data().asInt()));
        assertArrayEquals(new int[] {6, 6, 981668463, 1, 655372}, enc.data().asInt());

        assertEquals(3, elements);

        assertEquals(exp_0, initial);

        INDArray target = Nd4j.create(6);

        Nd4j.getExecutioner().bitmapDecode(enc, target);
        log.info("Target: {}", Arrays.toString(target.data().asFloat()));
        assertEquals(exp_1, target);
    }



    @Test
    public void testBitmapEncoding4() {
        Nd4j.getRandom().setSeed(119);
        INDArray initial = Nd4j.rand(1, 10000, 0, 1, Nd4j.getRandom());
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-1);

        Nd4j.getExecutioner().bitmapDecode(enc, initial);

        assertEquals(exp_1, initial);
    }

    @Test
    public void testBitmapEncoding5() {
        Nd4j.getRandom().setSeed(119);
        INDArray initial = Nd4j.rand(1, 10000, -1, -0.5, Nd4j.getRandom());
        INDArray exp_0 = initial.dup().addi(1e-1);
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-1);
        assertEquals(exp_0, initial);

        Nd4j.getExecutioner().bitmapDecode(enc, initial);

        assertEquals(exp_1, initial);
    }

    @Test
    public void testBitmapEncoding6() {
        Nd4j.getRandom().setSeed(119);
        INDArray initial = Nd4j.rand(1, 100000, -1, 1, Nd4j.getRandom());
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);
        //assertEquals(exp_0, initial);

        Nd4j.getExecutioner().bitmapDecode(enc, initial);

        assertEquals(exp_1, initial);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
