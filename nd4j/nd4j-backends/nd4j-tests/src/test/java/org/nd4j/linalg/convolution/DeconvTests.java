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

package org.nd4j.linalg.convolution;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;


import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
@NativeTag
@Tag(TagNames.FILE_IO)
public class DeconvTests extends BaseNd4jTestWithBackends {

    @TempDir Path testDir;

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LARGE_RESOURCES)
    public void compareKeras(Nd4jBackend backend) throws Exception {
        File newFolder = testDir.toFile();
        new ClassPathResource("keras/deconv/").copyDirectory(newFolder);

        File[] files = newFolder.listFiles();

        Set<String> tests = new HashSet<>();
        for(File file : files){
            String n = file.getName();
            if(!n.startsWith("mb"))
                continue;

            int idx = n.lastIndexOf('_');
            String name = n.substring(0, idx);
            tests.add(name);
        }

        List<String> l = new ArrayList<>(tests);
        Collections.sort(l);
        assertFalse(l.isEmpty());

        for(String s : l){
            String s2 = s.replaceAll("[a-zA-Z]", "");
            String[] nums = s2.split("_");
            int mb = Integer.parseInt(nums[0]);
            int k = Integer.parseInt(nums[1]);
            int size = Integer.parseInt(nums[2]);
            int stride = Integer.parseInt(nums[3]);
            boolean same = s.contains("same");
            int d = Integer.parseInt(nums[5]);
            boolean nchw = s.contains("nchw");

            INDArray w = Nd4j.readNpy(new File(newFolder, s + "_W.npy"));
            INDArray b = Nd4j.readNpy(new File(newFolder, s + "_b.npy"));
            INDArray in = Nd4j.readNpy(new File(newFolder, s + "_in.npy")).castTo(DataType.FLOAT);
            INDArray expOut = Nd4j.readNpy(new File(newFolder, s + "_out.npy"));

            CustomOp op = DynamicCustomOp.builder("deconv2d")
                    .addInputs(in, w, b)
                    .addIntegerArguments(
                            k, k,
                            stride, stride,
                            0, 0,   //padding
                            d, d,
                            same ? 1 : 0,
                            nchw ? 0 : 1)
                    .callInplace(false)
                    .build();
            INDArray out = Nd4j.create(op.calculateOutputShape().get(0));
            out.assign(Double.NaN);
            op.addOutputArgument(out);
            Nd4j.exec(op);

            boolean eq = expOut.equalsWithEps(out, 1e-4);
            assertTrue(eq);
        }
    }
}
