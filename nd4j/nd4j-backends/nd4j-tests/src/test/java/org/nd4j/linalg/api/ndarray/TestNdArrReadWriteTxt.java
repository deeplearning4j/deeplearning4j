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

package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.ArrayUtils;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j

public class TestNdArrReadWriteTxt extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void compareAfterWrite(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        int [] ranksToCheck = new int[] {0,1,2,3,4};
        for (int i = 0; i < ranksToCheck.length; i++) {
//            log.info("Checking read write arrays with rank " + ranksToCheck[i]);
            compareArrays(ranksToCheck[i],ordering(), testDir);
        }
    }

    public static void compareArrays(int rank, char ordering, Path testDir) throws Exception {
        List<Pair<INDArray, String>> all = NDArrayCreationUtil.getTestMatricesWithVaryingShapes(rank,ordering, Nd4j.defaultFloatingPointType());
        Iterator<Pair<INDArray,String>> iter = all.iterator();
        int cnt = 0;
        while (iter.hasNext()) {
            File dir = testDir.toFile();
            Pair<INDArray,String> currentPair = iter.next();
            INDArray origArray = currentPair.getFirst();
            //adding elements outside the bounds where print switches to scientific notation
            origArray.tensorAlongDimension(0,0).muli(0).addi(100000);
            origArray.putScalar(0,10001.1234);
//            log.info("\nChecking shape ..." + currentPair.getSecond());
            //log.info("C:\n"+ origArray.dup('c').toString());
//            log.info("F:\n"+ origArray.toString());
            Nd4j.writeTxt(origArray, new File(dir, "someArr.txt").getAbsolutePath());
            INDArray readBack = Nd4j.readTxt(new File(dir, "someArr.txt").getAbsolutePath());
            assertEquals(origArray, readBack,"\nNot equal on shape " + ArrayUtils.toString(origArray.shape()));
            cnt++;
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNd4jReadWriteText(@TempDir Path testDir,Nd4jBackend backend) throws Exception {

        File dir = testDir.toFile();
        int count = 0;
        for(val testShape : new long[][]{{1,1}, {3,1}, {4,5}, {1,2,3}, {2,1,3}, {2,3,1}, {2,3,4}, {1,2,3,4}, {2,3,4,2}}){
            List<Pair<INDArray, String>> l = null;
            switch (testShape.length){
                case 2:
                    l = NDArrayCreationUtil.getAllTestMatricesWithShape(testShape[0], testShape[1], 12345, Nd4j.defaultFloatingPointType());
                    break;
                case 3:
                    l = NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, testShape, Nd4j.defaultFloatingPointType());
                    break;
                case 4:
                    l = NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, testShape, Nd4j.defaultFloatingPointType());
                    break;
                default:
                    throw new RuntimeException();
            }


            for (Pair<INDArray, String> p : l) {
                File f = new File(dir, (count++) + ".txt");
                Nd4j.writeTxt(p.getFirst(), f.getAbsolutePath());

                INDArray read = Nd4j.readTxt(f.getAbsolutePath());
                String s = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
//                System.out.println(s);

                assertEquals(p.getFirst(), read);
            }
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
