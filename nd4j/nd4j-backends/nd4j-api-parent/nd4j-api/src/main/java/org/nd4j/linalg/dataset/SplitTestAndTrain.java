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

package org.nd4j.linalg.dataset;

import java.io.Serializable;

/**
 * SplitV test and train
 *
 * @author Adam Gibson
 */
public class SplitTestAndTrain implements Serializable {

    private DataSet train, test;

    public SplitTestAndTrain(DataSet train, DataSet test) {
        this.train = train;
        this.test = test;
    }

    public DataSet getTest() {
        return test;
    }

    public void setTest(DataSet test) {
        this.test = test;
    }

    public DataSet getTrain() {
        return train;
    }

    public void setTrain(DataSet train) {
        this.train = train;
    }
}
