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
 * Enumeration that is used for specifying the behaviour of repartitioning in {@link org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster}
 * (and possibly elsewhere.
 *
 * "Never" and "Always" repartition options are as expected; the "NumPartitionsWorkersDiffers" will repartition data if and only
 * if the number of partitions is not equal to the number of workers (total cores). Note however that even if the number of partitions
 * and number of workers differ, this does <i>not</i> guarantee that those partitions are balanced (in terms of number of
 * elements) in any way.
 *
 * @author Alex Black
 */
public enum Repartition {
    Never, Always, NumPartitionsWorkersDiffers
}
