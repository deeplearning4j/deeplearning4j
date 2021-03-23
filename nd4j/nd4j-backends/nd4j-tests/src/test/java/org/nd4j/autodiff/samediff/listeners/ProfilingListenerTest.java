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

package org.nd4j.autodiff.samediff.listeners;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.listeners.profiler.ProfilingListener;
import org.nd4j.autodiff.listeners.profiler.comparison.ProfileAnalyzer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class ProfilingListenerTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testProfilingListenerSimple(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 3);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 1, 2);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 2));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 2));
        SDVariable sm = sd.nn.softmax("predictions", in.mmul("matmul", w).add("addbias", b));
        SDVariable loss = sd.loss.logLoss("loss", label, sm);

        INDArray i = Nd4j.rand(DataType.FLOAT, 1, 3);
        INDArray l = Nd4j.rand(DataType.FLOAT, 1, 2);

        Path testDir = Paths.get(new File(System.getProperty("java.io.tmpdir")).toURI());
        File dir = testDir.resolve("new-dir-" + UUID.randomUUID().toString()).toFile();
        dir.mkdirs();
        File f = new File(dir, "test.json");
        f.deleteOnExit();
        ProfilingListener listener = ProfilingListener.builder(f)
                .recordAll()
                .warmup(5)
                .build();

        sd.setListeners(listener);
        Map<String,INDArray> ph = new HashMap<>();
        ph.put("in", i);

        for( int x = 0; x < 10; x++) {
            sd.outputSingle(ph, "predictions");
        }

        String content = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
//        System.out.println(content);
        assertFalse(content.isEmpty());
        //Should be 2 begins and 2 ends for each entry
        //5 warmup iterations, 5 profile iterations, x2 for both the op name and the op "instance" name
        String[] opNames = {"matmul", "add", "softmax"};
        for(String s : opNames) {
            assertEquals( 10, StringUtils.countMatches(content, s),s);
        }

        System.out.println("///////////////////////////////////////////");
        //ProfileAnalyzer.summarizeProfile(f, ProfileAnalyzer.ProfileFormat.SAMEDIFF);

    }
}
