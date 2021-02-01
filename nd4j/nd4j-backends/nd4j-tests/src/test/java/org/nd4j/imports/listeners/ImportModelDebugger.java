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
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Small utility for debugging Tensorflow import.
 *
 * If you get a java.lang.NoSuchMethodError: java.nio.ByteBuffer.rewind() run with Java 9 or higher
 *
 * Quick and dirty Python code to generate test data: (this is slow and could no doubt be improved)
 *
 * Also available in TFOpsTests via ZooEvaluation#write_intermediates
 *
 * <pre>
 * {@code
 * import os
 * import tensorflow as tf
 * import numpy as np
 *
 * # Read graph definition
 * with tf.gfile.GFile('C:\\Temp\\TF_Graphs\\faster_rcnn_resnet101_coco_2018_01_28\\frozen_inference_graph.pb', 'rb') as f:
 *     graph_def = tf.GraphDef()
 *     graph_def.ParseFromString(f.read())
 *
 * # Define input
 * inData = np.zeros([1,600,600,3], np.float32)
 * feed_dict = {"image_tensor:0":inData}
 * root_path = "C:\\Temp\\TF_Test\\"
 * path_sep = "\\"        #Change this for Linux to "/" (or comment out path.replace below)
 *
 * # Save input
 * for inName in feed_dict:
 *     inPath = root_path + "__placeholders" + path_sep + inName.replace(":", "__") + ".npy"
 *     parent_dir = os.path.dirname(inPath)
 *     os.makedirs(parent_dir, exist_ok=True)
 *     np.save(inPath, feed_dict[inName])
 *
 *
 * # now build the graph
 * with tf.Graph().as_default() as graph:
 *     tf.import_graph_def(graph_def, name="")
 *     first = True
 *     for op in graph.get_operations():
 *         print(op.name)
 *         #print("  ", op.outputs)
 *         sess = tf.Session()
 *         nOuts = len(op.outputs)
 *         for i in range(nOuts):
 *             try:
 *                 out = sess.run([op.name + ":" + str(i)], feed_dict=feed_dict)
 *                 path = root_path + op.name + "__" + str(i) + ".npy"
 *                 path = path.replace("/", path_sep)
 *                 parent_dir = os.path.dirname(path)
 *                 os.makedirs(parent_dir, exist_ok=True)
 *                 np.save(path, out[0])
 *             except Exception:
 *                 print("Error saving " + op.name + ":" + str(i))
 *                 traceback.print_exc()
 *                 print("-------------------------------------------------------------")
 * }
 * </pre>
 *
 * @author Alex Black
 */
@Ignore
public class ImportModelDebugger {

    @Test
    @Ignore
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
