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

package org.deeplearning4j.ui.play;

import org.deeplearning4j.ui.api.UIServer;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.api.buffer.DataType;

import java.io.File;

@Ignore
public class TestSameDiffUI {

    @Ignore
    @Test
    public void testSameDiff() throws Exception {

        File f = new File("C:/Temp/SameDiffUI/ui_data.bin");
        f.getParentFile().mkdirs();
        f.delete();

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 3);
        SDVariable w = sd.var("w", DataType.FLOAT, 3,4);
        SDVariable b = sd.var("b", DataType.FLOAT, 1, 4);

        SDVariable z = in.mmul(w).add(b);
        SDVariable a = sd.tanh(z);

        LogFileWriter lfw = new LogFileWriter(f);
        lfw.writeGraphStructure(sd);
        lfw.writeFinishStaticMarker();

        UIServer uiServer = UIServer.getInstance();


        Thread.sleep(1_000_000_000);
    }

}
