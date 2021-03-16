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

package org.nd4j.imports.listeners;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

@Disabled
public class ImportModelDebugger {

    @Test
    @Disabled
    public void doTest(){
        main(new String[0]);
    }

    public static void main(String[] args) {

        File modelFile = new File("C:\\Temp\\TF_Graphs\\cifar10_gan_85\\tf_model.pb");
        File rootDir = new File("C:\\Temp\\TF_Graphs\\cifar10_gan_85");

        SameDiff sd = TFGraphMapper.importGraph(modelFile);

        ImportDebugListener l = ImportDebugListener.builder(rootDir)
                .checkShapesOnly(true)
                .floatingPointEps(1e-5)
                .onFailure(ImportDebugListener.OnFailure.EXCEPTION)
                .logPass(true)
                .build();

        sd.setListeners(l);

        Map<String,INDArray> ph = loadPlaceholders(rootDir);

        List<String> outputs = sd.outputs();

        sd.output(ph, outputs);
    }


    public static Map<String, INDArray> loadPlaceholders(File rootDir){
        File dir = new File(rootDir, "__placeholders");
        if(!dir.exists()){
            throw new IllegalStateException("Cannot find placeholders: directory does not exist: " + dir.getAbsolutePath());
        }

        Map<String, INDArray> ret = new HashMap<>();
        Iterator<File> iter = FileUtils.iterateFiles(dir, null, true);
        while(iter.hasNext()){
            File f = iter.next();
            if(!f.isFile())
                continue;
            String s = dir.toURI().relativize(f.toURI()).getPath();
            int idx = s.lastIndexOf("__");
            String name = s.substring(0, idx);
            INDArray arr = Nd4j.createFromNpyFile(f);
            ret.put(name, arr);
        }

        return ret;
    }
}
