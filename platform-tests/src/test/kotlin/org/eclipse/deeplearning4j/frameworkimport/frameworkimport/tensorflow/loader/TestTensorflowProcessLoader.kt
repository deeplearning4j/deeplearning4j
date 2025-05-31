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

package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.tensorflow.loader

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.registry
import org.nd4j.samediff.frameworkimport.tensorflow.process.TensorflowMappingProcessLoader
import org.tensorflow.framework.*

@Tag(TagNames.TENSORFLOW)
class TestTensorflowProcessLoader {

    @Test
    fun testLoader() {
        val tensorflowOpMappingRegistry = OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>(
            "tensorflow", OpDescriptorLoaderHolder.nd4jOpDescriptor)

        val loader = TensorflowMappingProcessLoader(tensorflowOpMappingRegistry)
        println(loader)
        registry().inputFrameworkOpNames().forEach { name ->
            if(registry().hasMappingOpProcess(name)) {
                val process = registry().lookupOpMappingProcess(name)
                val serialized = process.serialize()
                val created = loader.createProcess(serialized)
                assertEquals(process,created,"Op name $name failed with process tensor rules ${process.tensorMappingRules()} and created tensor rules ${created.tensorMappingRules()} with attributes ${process.attributeMappingRules()} and created attribute rules ${created.attributeMappingRules()}")
            }

        }

    }




    @Test
    fun saveTest() {
        registry().saveProcessesAndRuleSet()
    }

}
