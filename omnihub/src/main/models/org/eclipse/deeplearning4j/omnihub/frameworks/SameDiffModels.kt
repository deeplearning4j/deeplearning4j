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

import org.eclipse.deeplearning4j.omnihub.OmniHubUtils
import org.eclipse.deeplearning4j.omnihub.dsl.FrameworkNamespace
import org.eclipse.deeplearning4j.omnihub.dsl.SameDiffModel

fun SameDiffModels() = FrameworkNamespace("SameDiffModels") {
    SameDiffModel("ageGooglenet","This model is converted from the onnx model age_googlenet.onnx at the onnx model zoo at: https://github.com/onnx/models","${OmniHubUtils.DEFAULT_OMNIHUB_URL}/samediff/age_googlenet.fb",true)
    SameDiffModel("resnet18","This model is converted from pytorch's torchvision resnet18 model via onnx. For more information see: https://pytorch.org/vision/stable/models.html","${OmniHubUtils.DEFAULT_OMNIHUB_URL}/samediff/resnet18.fb",true)

}