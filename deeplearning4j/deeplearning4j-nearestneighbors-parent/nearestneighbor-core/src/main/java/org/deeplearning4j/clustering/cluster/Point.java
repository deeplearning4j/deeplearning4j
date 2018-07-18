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

package org.deeplearning4j.clustering.cluster;

import lombok.AccessLevel;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 *
 */
@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Point implements Serializable {

    private static final long serialVersionUID = -6658028541426027226L;

    private String id = UUID.randomUUID().toString();
    private String label;
    private INDArray array;


    /**
     *
     * @param array
     */
    public Point(INDArray array) {
        super();
        this.array = array;
    }

    /**
     *
     * @param id
     * @param array
     */
    public Point(String id, INDArray array) {
        super();
        this.id = id;
        this.array = array;
    }

    public Point(String id, String label, double[] data) {
        this(id, label, Nd4j.create(data));
    }

    public Point(String id, String label, INDArray array) {
        super();
        this.id = id;
        this.label = label;
        this.array = array;
    }


    /**
     *
     * @param matrix
     * @return
     */
    public static List<Point> toPoints(INDArray matrix) {
        List<Point> arr = new ArrayList<>(matrix.rows());
        for (int i = 0; i < matrix.rows(); i++) {
            arr.add(new Point(matrix.slice(i)));
        }

        return arr;
    }

    /**
     *
     * @param vectors
     * @return
     */
    public static List<Point> toPoints(List<INDArray> vectors) {
        List<Point> points = new ArrayList<>();
        for (INDArray vector : vectors)
            points.add(new Point(vector));
        return points;
    }


}
