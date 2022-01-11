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
package org.nd4j.samediff.frameworkimport.onnx;

import onnx.Onnx;
import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.file.Path;

public class TestOnnxConverter {

    @TempDir
    Path tempDir;

    @Test
    public void test() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("mnist.onnx");
        File f = classPathResource.getFile();
        OnnxConverter onnxConverter = new OnnxConverter();
        Onnx.ModelProto modelProto = Onnx.ModelProto.parseFrom(new FileInputStream(f));
        Onnx.GraphProto graphProto = onnxConverter.addConstValueInfoToGraph(modelProto.getGraph());
        modelProto = modelProto.toBuilder().setGraph(graphProto).build();
        File newModel = new File(tempDir.toFile(),"postprocessed.onnx");
        IOUtils.write(modelProto.toByteArray(),new FileOutputStream(newModel));
        File file = new File(tempDir.toFile(), "newfile.onnx");
        onnxConverter.convertModel(newModel,file);
    }





}
