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

package org.deeplearning4j.clustering.sptree;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 *
 * A vector with an index and function for distance
 * @author Adam Gibson
 */
@Data
public class DataPoint implements Serializable {
    private int index;
    private INDArray point;
    private long d;
    private String functionName;
    private boolean invert = false;


    public DataPoint(int index, INDArray point, boolean invert) {
        this(index, point, "euclidean");
        this.invert = invert;
    }

    public DataPoint(int index, INDArray point, String functionName, boolean invert) {
        this.index = index;
        this.point = point;
        this.functionName = functionName;
        this.d = point.length();
        this.invert = invert;
    }


    public DataPoint(int index, INDArray point) {
        this(index, point, false);
    }

    public DataPoint(int index, INDArray point, String functionName) {
        this(index, point, functionName, false);
    }

    /**
     * Euclidean distance
     * @param point the distance from this point to the given point
     * @return the distance between the two points
     */
    public float distance(DataPoint point) {
        switch (functionName) {
            case "euclidean":
                float ret = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(this.point, point.point))
                                .getFinalResult().floatValue();
                return invert ? -ret : ret;

            case "cosinesimilarity":
                float ret2 = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(this.point, point.point))
                                .getFinalResult().floatValue();
                return invert ? -ret2 : ret2;

            case "manhattan":
                float ret3 = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(this.point, point.point))
                                .getFinalResult().floatValue();
                return invert ? -ret3 : ret3;
            case "dot":
                float dotRet = (float) Nd4j.getBlasWrapper().dot(this.point, point.point);
                return invert ? -dotRet : dotRet;
            default:
                float ret4 = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(this.point, point.point))
                                .getFinalResult().floatValue();
                return invert ? -ret4 : ret4;

        }
    }

}
