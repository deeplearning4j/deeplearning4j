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

package org.deeplearning4j.spark.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * A simple configuration object (common settings for workers)
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class WorkerConfiguration implements Serializable {

    protected final boolean isGraphNetwork;
    protected final int dataSetObjectSizeExamples; //Number of examples in each DataSet object
    protected final int batchSizePerWorker;
    protected final int maxBatchesPerWorker;
    protected final int prefetchNumBatches;
    protected final boolean collectTrainingStats;

}
