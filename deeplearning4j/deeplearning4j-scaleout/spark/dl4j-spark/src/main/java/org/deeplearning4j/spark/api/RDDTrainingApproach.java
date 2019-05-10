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

/**
 * Approach to use when training from a {@code JavaRDD<DataSet>} or {@code JavaRDD<MultiDataSet>}.
 *
 * <b>Export</b>: first export the RDD to disk (temporary directory) and train from that.
 * <b>Direct</b>: aka 'legacy mode': train directly from the RDD. This has higher memory requirements and lower performance
 *  compared to the Export approach. It does not export the data to disk first, hence uses less space.
 *
 * @author Alex Black
 */
public enum RDDTrainingApproach {
    Export, Direct
}
