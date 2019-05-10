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

package org.nd4j.linalg.heartbeat.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.heartbeat.reports.Task;

/**
 * @author raver119@gmail.com
 */
public class TaskUtils {
    private TaskUtils() {}

    public static Task buildTask(INDArray[] array, INDArray[] labels) {
        Task task = new Task();

        return task;
    }

    public static Task buildTask(INDArray array, INDArray labels) {
        return new Task();
    }

    public static Task buildTask(INDArray array) {
        return new Task();
    }

    public static Task buildTask(DataSet dataSet) {
        return new Task();
    }

    public static Task buildTask(org.nd4j.linalg.dataset.api.DataSet dataSet) {
        return new Task();
    }

    public static Task buildTask(DataSetIterator dataSetIterator) {
        return new Task();
    }
}
