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

package org.nd4j.linalg.indexing;

/**
 * @author Adam Gibson
 */
public class IndexInfo {
    private INDArrayIndex[] indexes;
    private boolean[] point;
    private boolean[] newAxis;
    private int numNewAxes = 0;
    private int numPoints = 0;

    public IndexInfo(INDArrayIndex... indexes) {
        this.indexes = indexes;
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] instanceof PointIndex)
                numPoints++;
            if (indexes[i] instanceof IntervalIndex) {

            }
            if (indexes[i] instanceof NewAxis)
                numNewAxes++;
        }

    }

    public int getNumNewAxes() {
        return numNewAxes;
    }

    public int getNumPoints() {
        return numPoints;
    }
}
