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
package org.eclipse.deeplearning4j.omnihub.dsl

import org.eclipse.deeplearning4j.omnihub.api.FrameworkNamespace
import org.eclipse.deeplearning4j.omnihub.api.Model
import org.eclipse.deeplearning4j.omnihub.api.ModelType
import org.eclipse.deeplearning4j.omnihub.api.NamespaceModels

fun FrameworkNamespace(name: String, block: NamespaceModels.() -> Unit): NamespaceModels {
    var ret = NamespaceModels(name)
    ret.block()
    return ret
}




fun NamespaceModels.DL4JModel(name: String,
                              documentation: String,
                              url: String,
                              pretrained:Boolean = true,
                              modelType: ModelType,
                              block: (Model.() -> Unit)? = null): Model {
    return this.Model(name,documentation,url,pretrained,FrameworkNamespace.DL4J,modelType,block)


}


fun NamespaceModels.SameDiffModel(name: String,
                                  documentation: String,
                                  url: String,
                                  pretrained:Boolean = true,
                                  block: (Model.() -> Unit)? = null): Model {
    return this.Model(name,documentation,url,pretrained,FrameworkNamespace.SAMEDIFF,ModelType.COMP_GRAPH,block)


}

fun NamespaceModels.Model(name: String,
                          documentation: String,
                          url: String,
                          pretrained:Boolean = true,
                          framework: FrameworkNamespace,
                          modelType: ModelType = ModelType.COMP_GRAPH,
                          block: (Model.() -> Unit)? = null): Model {
    var model = org.eclipse.deeplearning4j.omnihub.api.Model(url,name,pretrained,documentation,framework,modelType)
    if(block != null)
        model.block()
    this.models.add(model)
    return model


}