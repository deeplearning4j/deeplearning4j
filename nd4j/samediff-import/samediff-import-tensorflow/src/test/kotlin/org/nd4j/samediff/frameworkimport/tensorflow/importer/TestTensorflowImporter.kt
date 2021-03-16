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

import junit.framework.Assert
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.nd4j.common.io.ClassPathResource

class TestTensorflowImporter {

    @Test
    @Disabled
    fun testImporter() {
        val tfFrameworkImport = TensorflowFrameworkImporter()
        val tfFile = ClassPathResource("lenet_frozen.pb").file
        val graph  = tfFrameworkImport.runImport(tfFile.absolutePath)
        //note this is just a test to make sure everything runs, we test the underlying import elsewhere
        Assert.assertNotNull(graph)
    }

}