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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.eclipse.deeplearning4j.omnihub.models;

import java.lang.Exception;
import org.nd4j.autodiff.samediff.SameDiff;

public class SameDiffModels {
  public SameDiffModels() {
  }

  /**
   * This model is converted from the onnx model age_googlenet.onnx at the onnx model zoo at: https://github.com/onnx/models */
  public SameDiff ageGooglenet(boolean forceDownload) throws Exception {
    return org.eclipse.deeplearning4j.omnihub.OmniHubUtils.loadSameDiffModel("age_googlenet.fb",forceDownload);
  }

  /**
   * This model is converted from pytorch's torchvision resnet18 model via onnx. For more information see: https://pytorch.org/vision/stable/models.html */
  public SameDiff resnet18(boolean forceDownload) throws Exception {
    return org.eclipse.deeplearning4j.omnihub.OmniHubUtils.loadSameDiffModel("resnet18.fb",forceDownload);
  }
}
