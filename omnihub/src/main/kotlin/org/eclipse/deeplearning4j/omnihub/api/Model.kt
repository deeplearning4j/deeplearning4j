/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.omnihub.api

interface ModelLike {
    fun modelUrl(): String
    fun modelName(): String
    fun pretrained(): Boolean
    fun documentation(): String
    fun framework(): FrameworkNamespace
    fun modelType(): ModelType

}

data class Model(val modelUrl: String,
                 val modelName: String,
                 val pretrained: Boolean,
                 val documentation: String,
                 val framework: FrameworkNamespace,
                 val modelType: ModelType = ModelType.COMP_GRAPH): ModelLike {
    override fun modelUrl() = modelUrl
    override fun modelName() = modelName
    override fun pretrained() = pretrained
    override fun documentation() = documentation
    override fun framework() = framework
    override fun modelType() = modelType

}