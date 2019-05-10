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

package org.deeplearning4j.spark.models.sequencevectors.export;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This interface describes
 *
 * @author raver119@gmail.com
 */
public interface SparkModelExporter<T extends SequenceElement> {

    /**
     * This method will be called at final stage of SequenceVectors training, and JavaRDD being passed as argument will
     *
     * @param rdd
     */
    void export(JavaRDD<ExportContainer<T>> rdd);
}
