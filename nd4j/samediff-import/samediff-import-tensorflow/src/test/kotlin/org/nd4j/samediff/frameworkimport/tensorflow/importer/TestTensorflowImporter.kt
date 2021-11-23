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
package org.nd4j.samediff.frameworkimport.tensorflow.importer

import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Test
import org.nd4j.common.io.ClassPathResource
import org.nd4j.linalg.factory.Nd4j

class TestTensorflowImporter {

    @Test
    fun testImporter() {
        Nd4j.getExecutioner().enableDebugMode(true)
        Nd4j.getExecutioner().enableVerboseMode(true)
        val tfFrameworkImport = TensorflowFrameworkImporter()
        val tfFile = ClassPathResource("lenet_frozen.pb").file
        val graph  = tfFrameworkImport.runImport(tfFile.absolutePath,mapOf("input" to  Nd4j.ones(1,784)))
        //note this is just a test to make sure everything runs, we test the underlying import elsewhere
        assertNotNull(graph)
    }


}