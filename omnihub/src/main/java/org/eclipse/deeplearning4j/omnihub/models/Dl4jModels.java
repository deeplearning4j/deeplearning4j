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
import org.deeplearning4j.nn.graph.ComputationGraph;

public class Dl4jModels {
  public Dl4jModels() {
  }

  /**
   * This model is converted from keras applications.Keras applications vgg19 weights can be created with: */
  public ComputationGraph vgg19noTop(boolean forceDownload) throws Exception {
    return org.eclipse.deeplearning4j.omnihub.OmniHubUtils.loadCompGraph("vgg19_weights_tf_dim_ordering_tf_kernels_notop.zip",forceDownload);
  }
}
