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
package org.eclipse.deeplearning4j.omnihub.frameworks

import org.deeplearning4j.omnihub.OmnihubConfig
import org.eclipse.deeplearning4j.omnihub.api.ModelType
import org.eclipse.deeplearning4j.omnihub.dsl.DL4JModel
import org.eclipse.deeplearning4j.omnihub.dsl.FrameworkNamespace

fun Dl4jModels() = FrameworkNamespace("DL4JModels") {
    DL4JModel("vgg19noTop","This model is converted from keras applications." +
            "Keras applications vgg19 weights can be created with:","${OmnihubConfig.DEFAULT_OMNIHUB_URL}/dl4j/vgg19_weights_tf_dim_ordering_tf_kernels_notop.zip",
        true,
        ModelType.COMP_GRAPH)
}