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

package org.nd4j.imports;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.OpValidationSuite;
import org.nd4j.imports.TFGraphs.TFGraphTestAllHelper;
import org.nd4j.imports.TFGraphs.TFGraphTestAllLibnd4j;
import org.nd4j.imports.TFGraphs.TFGraphTestAllSameDiff;
import org.nd4j.imports.TFGraphs.TFGraphTestZooModels;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * This set of tests suited for validation of various graph execuction methods: flatbuffers, stored graphs reuse, one-by-one execution, etc
 *
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class ExecutionTests extends BaseNd4jTest {

    public ExecutionTests(Nd4jBackend backend) {
        super(backend);
    }

    @After
    public void tearDown() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }


    @Test
    public void testStoredGraph_1()  throws Exception {
        if(TFGraphTestZooModels.isPPC()){
            /*
            Ugly hack to temporarily disable tests on PPC only on CI
            Issue logged here: https://github.com/deeplearning4j/deeplearning4j/issues/7657
            These will be re-enabled for PPC once fixed - in the mean time, remaining tests will be used to detect and prevent regressions
             */
            log.warn("TEMPORARILY SKIPPING TEST ON PPC ARCHITECTURE DUE TO KNOWN JVM CRASH ISSUES - SEE https://github.com/deeplearning4j/deeplearning4j/issues/7657");
            OpValidationSuite.ignoreFailing();
        }

        Nd4j.create(1);

        val tg = TFGraphMapper.importGraphTxt(new ClassPathResource("tf_graphs/reduce_dim.pb.txt").getInputStream(), null, null);
        System.out.println(tg.summary());

        Map<String,INDArray> result_0 = tg.exec(Collections.emptyMap(), tg.outputs());
        val exp_0 = Nd4j.create(DataType.FLOAT, 3).assign(3.0);

        assertEquals(exp_0, result_0.get("Sum"));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
