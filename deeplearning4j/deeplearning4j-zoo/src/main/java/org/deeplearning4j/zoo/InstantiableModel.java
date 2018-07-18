/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.zoo;

import org.deeplearning4j.nn.api.Model;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 */
public interface InstantiableModel {

    void setInputShape(int[][] inputShape);

    <M extends Model> M init();

    @Deprecated ModelMetaData metaData();

    Class<? extends Model> modelType();

    String pretrainedUrl(PretrainedType pretrainedType);

    long pretrainedChecksum(PretrainedType pretrainedType);

    String modelName();
}
