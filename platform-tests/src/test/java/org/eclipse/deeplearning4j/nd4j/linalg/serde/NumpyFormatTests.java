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

package org.eclipse.deeplearning4j.nd4j.linalg.serde;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Disabled;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Tag(TagNames.FILE_IO)
@Tag(TagNames.NDARRAY_SERDE)
public class NumpyFormatTests extends BaseNd4jTestWithBackends {

    @TempDir Path testDir;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToNpyFormat(Nd4jBackend backend) throws Exception {
        val dir = testDir.resolve("new-dir-" + UUID.randomUUID()).toFile();
        assertTrue(dir.mkdirs());
        new ClassPathResource("numpy_arrays/").copyDirectory(dir);

        File[] files = dir.listFiles();
        int cnt = 0;

        for(File f : files) {
            if(!f.getPath().endsWith(".npy")) {
                log.warn("Skipping: {}", f);
                continue;
            }

            String path = f.getAbsolutePath();
            int lastDot = path.lastIndexOf('.');
            int lastUnderscore = path.lastIndexOf('_');
            String dtype = path.substring(lastUnderscore + 1, lastDot);

            DataType dt = DataType.fromNumpy(dtype);

            INDArray arr = Nd4j.arange(12).castTo(dt).reshape(3,4);
            arr.dataType();
            byte[] bytes = Nd4j.toNpyByteArray(arr);
            //strip out null terminated characters, most runtimes ignore this
            byte[] expected = FileUtils.readFileToByteArray(f);
            String bytesString = new String(bytes);
            String expectedString = new String(expected);
            String bytesReplaceNul = bytesString.replaceAll("\0","");
            String expectedReplaceNull = expectedString.replaceAll("\0","");
            INDArray resultArr = Nd4j.createNpyFromByteArray(bytes);
            INDArray expectedArr = Nd4j.createNpyFromByteArray(expected);
            assertEquals(expectedArr, resultArr,"Failed with file [" + f.getName() + "] on data type " + dt);
            cnt++;
        }

        assertTrue(cnt > 0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToNpyFormatScalars(Nd4jBackend backend) throws Exception {
        val dir = testDir.resolve("new-path0" + UUID.randomUUID()).toFile();
        dir.mkdirs();
        new ClassPathResource("numpy_arrays/scalar/").copyDirectory(dir);

        File[] files = dir.listFiles();
        int cnt = 0;

        for(File f : files) {
            if(!f.getPath().endsWith(".npy")) {
                log.warn("Skipping: {}", f);
                continue;
            }

            String path = f.getAbsolutePath();
            int lastDot = path.lastIndexOf('.');
            int lastUnderscore = path.lastIndexOf('_');
            String dtype = path.substring(lastUnderscore + 1, lastDot);

            DataType dt = DataType.fromNumpy(dtype);

            INDArray arr = Nd4j.scalar(dt, 1);
            byte[] bytes = Nd4j.toNpyByteArray(arr);
            byte[] expected = FileUtils.readFileToByteArray(f);
            assertEquals(Nd4j.createNpyFromByteArray(bytes),Nd4j.createNpyFromByteArray(expected));
            cnt++;

        }

        assertTrue(cnt > 0);
    }



    @Test
    public void testNumpyConversion() throws Exception {
        INDArray linspace = Nd4j.linspace(1,4,4, DataType.FLOAT);
        DataBuffer convertBuffer = Nd4j.getNDArrayFactory().convertToNumpyBuffer(linspace);
        Pointer convert = Nd4j.getNDArrayFactory().convertToNumpy(linspace);
        Pointer pointer = NativeOpsHolder.getInstance().getDeviceNativeOps().loadNpyFromHeader(convert);
        Pointer pointer1 = NativeOpsHolder.getInstance().getDeviceNativeOps().dataPointForNumpyStruct(pointer);
        pointer1.capacity(linspace.data().getElementSize() * linspace.data().length());
        ByteBuffer byteBuffer = linspace.data().pointer().asByteBuffer();
        byte[] originalData = new byte[byteBuffer.capacity()];
        byteBuffer.get(originalData);


        ByteBuffer floatBuffer = pointer1.asByteBuffer();
        byte[] dataTwo = new byte[floatBuffer.capacity()];
        floatBuffer.get(dataTwo);
        assertArrayEquals(originalData,dataTwo);
        Buffer buffer = (Buffer) floatBuffer;
        buffer.position(0);

    }

    @Test
    @Disabled("Test is very large compared to most tests. It needs to be to test the limits of memcpy/heap space.")
    public void testLargeNumpyWrite() throws Exception {
        Arrays.stream(DataType.values()).filter(input ->
                        input != DataType.BFLOAT16 && input != DataType.COMPRESSED && input != DataType.UTF8)
                .forEach(dataType -> {
                    System.out.println("Trying with data type " + dataType);
                    INDArray largeArr = Nd4j.create(dataType,115240, 2400);

                    File tempFile = new File("large-npy-" + dataType.name() + ".npy");
                    tempFile.deleteOnExit();
                    try {
                        Nd4j.writeAsNumpy(largeArr,tempFile);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    assertTrue(tempFile.exists());
                    INDArray read = Nd4j.createFromNpyFile(tempFile);
                    assertEquals(largeArr,read);
                });

    }



    @Test
    public void testNumpyWrite() throws Exception {
        INDArray linspace = Nd4j.linspace(1,4,4, Nd4j.dataType());
        File tmpFile = new File(System.getProperty("java.io.tmpdir"),"nd4j-numpy-tmp-" + UUID.randomUUID().toString() + ".bin");
        tmpFile.deleteOnExit();
        Nd4j.writeAsNumpy(linspace,tmpFile);

        INDArray numpyFromFile = Nd4j.createFromNpyFile(tmpFile);
        assertEquals(linspace,numpyFromFile);
    }

    @Test
    public void testNpyByteArray() throws Exception {
        INDArray linspace = Nd4j.linspace(1,4,4, Nd4j.dataType());
        byte[] bytes = Nd4j.toNpyByteArray(linspace);
        INDArray fromNpy = Nd4j.createNpyFromByteArray(bytes);
        assertEquals(linspace,fromNpy);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNpzReading(Nd4jBackend backend) throws Exception {

        val dir = testDir.resolve("new-folder-npz").toFile();
        dir.mkdirs();
        new ClassPathResource("numpy_arrays/npz/").copyDirectory(dir);

        File[] files = dir.listFiles();
        int cnt = 0;

        for(File f : files){
            if(!f.getPath().endsWith(".npz")){
                log.warn("Skipping: {}", f);
                continue;
            }

            String path = f.getAbsolutePath();
            int lastDot = path.lastIndexOf('.');
            int lastSlash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
            String dtype = path.substring(lastSlash + 1, lastDot);

            DataType dt = DataType.fromNumpy(dtype);

            INDArray arr = Nd4j.arange(12).castTo(dt).reshape(3,4);
            INDArray arr2 = Nd4j.createFromArray(new float[]{0.00f,10.0f,20.0f});

            Map<String,INDArray> m = Nd4j.createFromNpzFile(f);
            assertEquals(2, m.size());
            assertTrue(m.containsKey("firstArr"));
            assertTrue(m.containsKey("secondArr"));

            assertEquals(arr, m.get("firstArr"));
            assertEquals(arr2, m.get("secondArr"));
            cnt++;
        }

        assertTrue(cnt > 0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTxtReading(Nd4jBackend backend) throws Exception {
        File f = new ClassPathResource("numpy_arrays/txt/arange_3,4_float32.txt").getFile();
        INDArray arr = Nd4j.readNumpy(DataType.FLOAT, f.getPath());

        INDArray exp = Nd4j.arange(12).castTo(DataType.FLOAT).reshape(3,4);
        assertEquals(exp, arr);

        arr = Nd4j.readNumpy(DataType.DOUBLE, f.getPath());

        assertEquals(exp.castTo(DataType.DOUBLE), arr);

        f = new ClassPathResource("numpy_arrays/txt_tab/arange_3,4_float32.txt").getFile();
        arr = Nd4j.readNumpy(DataType.FLOAT, f.getPath(), "\t");

        assertEquals(exp, arr);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNpy(Nd4jBackend backend) throws Exception {
        for(boolean empty : new boolean[]{false, true}) {
            val dir = testDir.resolve("new-dir-1-" + UUID.randomUUID().toString()).toFile();
            assertTrue(dir.mkdirs());
            if(!empty) {
                new ClassPathResource("numpy_arrays/npy/3,4/").copyDirectory(dir);
            } else {
                new ClassPathResource("numpy_arrays/npy/0,3_empty/").copyDirectory(dir);
            }

            File[] files = dir.listFiles();
            int cnt = 0;

            for (File f : files) {
                if (!f.getPath().endsWith(".npy")) {
                    log.warn("Skipping: {}", f);
                    continue;
                }

                String path = f.getAbsolutePath();
                int lastDot = path.lastIndexOf('.');
                int lastUnderscore = path.lastIndexOf('_');
                String dtype = path.substring(lastUnderscore + 1, lastDot);
                DataType dt = DataType.fromNumpy(dtype);

                INDArray exp;
                if(empty){
                    exp = Nd4j.create(dt, 0, 3);
                } else {
                    exp = Nd4j.arange(12).castTo(dt).reshape(3, 4);
                }
                INDArray act = Nd4j.createFromNpyFile(f);

                assertEquals( exp, act,"Failed with file [" + f.getName() + "]");
                cnt++;
            }

            assertTrue(cnt > 0);
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void readNumpyCorruptHeader1(Nd4jBackend backend) throws Exception {
        assertThrows(RuntimeException.class,() -> {
            File f = testDir.toFile();

            File fValid = new ClassPathResource("numpy_arrays/arange_3,4_float32.npy").getFile();
            byte[] numpyBytes = FileUtils.readFileToByteArray(fValid);
            for( int i = 0; i < 10; i++) {
                numpyBytes[i] = 0;
            }
            File fCorrupt = new File(f, "corrupt.npy");
            FileUtils.writeByteArrayToFile(fCorrupt, numpyBytes);

            INDArray exp = Nd4j.arange(12).castTo(DataType.FLOAT).reshape(3,4);

            INDArray act1 = Nd4j.createFromNpyFile(fValid);
            assertEquals(exp, act1);

            INDArray probablyShouldntLoad = Nd4j.createFromNpyFile(fCorrupt); //Loads fine
            boolean eq = exp.equals(probablyShouldntLoad); //And is actually equal content
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void readNumpyCorruptHeader2(Nd4jBackend backend) throws Exception {
        assertThrows(RuntimeException.class,() -> {
            File f = testDir.toFile();

            File fValid = new ClassPathResource("numpy_arrays/arange_3,4_float32.npy").getFile();
            byte[] numpyBytes = FileUtils.readFileToByteArray(fValid);
            for( int i = 1; i < 10; i++) {
                numpyBytes[i] = 0;
            }
            File fCorrupt = new File(f, "corrupt.npy");
            FileUtils.writeByteArrayToFile(fCorrupt, numpyBytes);

            INDArray exp = Nd4j.arange(12).castTo(DataType.FLOAT).reshape(3,4);

            INDArray act1 = Nd4j.createFromNpyFile(fValid);
            assertEquals(exp, act1);

            INDArray probablyShouldntLoad = Nd4j.createFromNpyFile(fCorrupt); //Loads fine
            boolean eq = exp.equals(probablyShouldntLoad); //And is actually equal content
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAbsentNumpyFile_1(Nd4jBackend backend) throws Exception {
        assertThrows(IllegalArgumentException.class,() -> {
            val f = new File("pew-pew-zomg.some_extension_that_wont_exist");
            INDArray act1 = Nd4j.createFromNpyFile(f);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testAbsentNumpyFile_2(Nd4jBackend backend) throws Exception {
        assertThrows(IllegalArgumentException.class,() -> {
            val f = new File("c:/develop/batch-x-1.npy");
            INDArray act1 = Nd4j.createFromNpyFile(f);
            log.info("Array shape: {}; sum: {};", act1.shape(), act1.sumNumber().doubleValue());
        });

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
