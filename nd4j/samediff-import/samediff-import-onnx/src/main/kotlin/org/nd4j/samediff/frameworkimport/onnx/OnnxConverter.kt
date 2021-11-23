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
package org.nd4j.samediff.frameworkimport.onnx

import lombok.SneakyThrows
import onnx.Onnx
import onnx.Onnx.OperatorSetIdProto
import org.apache.commons.io.FileUtils
import org.apache.commons.io.IOUtils
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.onnx.*
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer

class OnnxConverter {

    @SneakyThrows
    fun convertModel(inputModel: File,outputModelFilePath: File)  {
        val converter = DefaultVersionConverter()
        val bytes = ByteBuffer.wrap(IOUtils.toByteArray(BufferedInputStream(FileInputStream(inputModel))))
        val bytePointer = BytePointer(bytes)
        val proto = ModelProto()
        //val operatorSet = Onnx.OperatorSetIdProto()
        proto.ParseFromString(bytePointer)
        val initialId = OpSetID(0)
        for(i in 0 until proto.opset_import_size()) {
            val opSetImport = proto.opset_import(i)
            if(!opSetImport.has_domain() || opSetImport.domain().string == "ai.onnx") {
                //approximates default opset from https://github.com/onnx/onnx/blob/master/onnx/version_converter/convert.cc#L14
                initialId.setVersion(opSetImport.version().toInt())
                break

            }
        }

        val convertVersion = converter.convert_version(proto, initialId, OpSetID(13))
        val save = convertVersion.SerializeAsString()
        IOUtils.write(save.stringBytes, FileOutputStream(outputModelFilePath))

    }




}