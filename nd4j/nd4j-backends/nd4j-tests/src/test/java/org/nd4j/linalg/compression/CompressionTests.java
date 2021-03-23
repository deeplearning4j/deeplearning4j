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

package org.nd4j.linalg.compression;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
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


import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
@Tag(TagNames.COMPRESSION)
public class CompressionTests extends BaseNd4jTestWithBackends {



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompressionDescriptorSerde(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGzipInPlaceCompression(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
        Nd4j.getCompressor().setDefaultCompression("GZIP");
        Nd4j.getCompressor().compressi(array);
        assertTrue(array.isCompressed());
        Nd4j.getCompressor().decompressi(array);
        assertFalse(array.isCompressed());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGzipCompression1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 10000, 20000, DataType.FLOAT);
        INDArray exp = array.dup();

        BasicNDArrayCompressor.getInstance().setDefaultCompression("GZIP");

        INDArray compr = BasicNDArrayCompressor.getInstance().compress(array);

        assertEquals(DataType.COMPRESSED, compr.data().dataType());

        INDArray decomp = BasicNDArrayCompressor.getInstance().decompress(compr);

        assertEquals(exp, array);
        assertEquals(exp, decomp);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoOpCompression1(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testJVMCompression3(Nd4jBackend backend) {
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


    @Disabled
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression0(Nd4jBackend backend) {
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
    @Disabled
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression1(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression2(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression3(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression4(Nd4jBackend backend) {
        INDArray initial = Nd4j.create(new double[] {1e-4, -1e-4, 0.0, 0.0, 1e-4, -1e-4});
        INDArray exp_0 = initial.dup();


        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1e-3f);


        log.info("Initial array: {}", Arrays.toString(initial.data().asFloat()));

        assertEquals(exp_0, initial);

        assertNull(compressed);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression5(Nd4jBackend backend) {
        INDArray initial = Nd4j.ones(10);
        INDArray exp_0 = initial.dup();

        Nd4j.getExecutioner().commit();

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1.0f, 3);

        assertEquals(7, compressed.data().length());

        assertNotEquals(exp_0, initial);

        assertEquals(7, initial.sumNumber().doubleValue(), 0.01);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression5_1(Nd4jBackend backend) {
        INDArray initial = Nd4j.ones(1000);
        INDArray exp_0 = initial.dup();

        Nd4j.getExecutioner().commit();

        //Nd4j.getCompressor().getCompressor("THRESHOLD").configure(1e-3);
        INDArray compressed = Nd4j.getExecutioner().thresholdEncode(initial, 1.0f, 100);

        assertEquals(104, compressed.data().length());

        assertNotEquals(exp_0, initial);

        assertEquals(900, initial.sumNumber().doubleValue(), 0.01);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdCompression6(Nd4jBackend backend) {
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



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testThresholdSerialization1(Nd4jBackend backend) throws Exception {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding1(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding1_1(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding2(Nd4jBackend backend) {
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding3(Nd4jBackend backend) {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        INDArray initial = Nd4j.create(new float[] {0.0f, -6e-4f, 1e-3f, -1e-3f, 0.0f, 0.0f});
        INDArray exp_0 = Nd4j.create(new float[] {0.0f, -1e-4f, 0.0f, 0.0f, 0.0f, 0.0f});
        INDArray exp_1 = Nd4j.create(new float[] {0.0f, -5e-4f, 1e-3f, -1e-3f, 0.0f, 0.0f});


        INDArray enc = Nd4j.create(DataType.INT32, initial.length() / 16 + 5);

        long elements = Nd4j.getExecutioner().bitmapEncode(initial, enc, 1e-3);
        log.info("Encoded: {}", Arrays.toString(enc.data().asInt()));
        assertArrayEquals(new int[] {6, 6, 981668463, 1, 655372}, enc.data().asInt());

        assertEquals(3, elements);

        assertEquals(exp_0, initial);

        INDArray target = Nd4j.create(6);

        Nd4j.getExecutioner().bitmapDecode(enc, target);
        log.info("Target: {}", Arrays.toString(target.data().asFloat()));
        assertEquals(exp_1, target.castTo(exp_1.dataType()));
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding4(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(119);
        INDArray initial = Nd4j.rand(new int[]{1, 10000}, 0, 1, Nd4j.getRandom());
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-1);

        Nd4j.getExecutioner().bitmapDecode(enc, initial);

        assertEquals(exp_1, initial);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding5(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(119);
        INDArray initial = Nd4j.rand(new int[]{10000}, -1, -0.5, Nd4j.getRandom());
        INDArray exp_0 = initial.dup().addi(1e-1);
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-1);
        assertEquals(exp_0, initial);

        Nd4j.getExecutioner().bitmapDecode(enc, initial);

        assertEquals(exp_1, initial);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBitmapEncoding6(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(119);
        INDArray initial = Nd4j.rand(new int[]{10000}, -1, 1, Nd4j.getRandom());
        INDArray exp_1 = initial.dup();

        INDArray enc = Nd4j.getExecutioner().bitmapEncode(initial, 1e-3);
        //assertEquals(exp_0, initial);

        Nd4j.getExecutioner().bitmapDecode(enc, initial);

        val f0 = exp_1.toFloatVector();
        val f1 = initial.toFloatVector();

        assertArrayEquals(f0, f1, 1e-5f);

        assertEquals(exp_1, initial);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
