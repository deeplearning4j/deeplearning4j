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

package org.deeplearning4j.clustering.kdtree;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 12/29/14.
 */
public class HyperRect implements Serializable {

    private List<Interval> points;

    public HyperRect(List<Interval> points) {
        this.points = points;
    }


    public void enlargeTo(INDArray point) {
        for (int i = 0; i < points.size(); i++)
            points.get(i).enlarge(point.getDouble(i));
    }


    public static List<Interval> point(INDArray vector) {
        List<Interval> ret = new ArrayList<>();
        for (int i = 0; i < vector.length(); i++) {
            double curr = vector.getDouble(i);
            ret.add(new Interval(curr, curr));
        }
        return ret;
    }


    public List<Boolean> contains(INDArray hPoint) {
        List<Boolean> ret = new ArrayList<>();
        for (int i = 0; i < hPoint.length(); i++)
            ret.add(points.get(i).contains(hPoint.getDouble(i)));
        return ret;
    }

    public double minDistance(INDArray hPoint) {
        double ret = 0.0;
        for (int i = 0; i < hPoint.length(); i++) {
            double p = hPoint.getDouble(i);
            Interval interval = points.get(i);
            if (p < interval.lower)
                ret += Math.pow((p - interval.lower), 2);
            else
                ret += Math.pow((p - interval.higher), 2);
        }

        ret = Math.pow(ret, 0.5);


        return ret;
    }

    public HyperRect getUpper(INDArray hPoint, int desc) {
        Interval interval = points.get(desc);
        double d = hPoint.getDouble(desc);
        if (interval.higher < d)
            return null;
        HyperRect ret = new HyperRect(new ArrayList<>(points));
        Interval i2 = ret.points.get(desc);
        if (i2.lower < d)
            i2.lower = d;
        return ret;
    }

    public HyperRect getLower(INDArray hPoint, int desc) {
        Interval interval = points.get(desc);
        double d = hPoint.getDouble(desc);
        if (interval.higher > d)
            return null;
        HyperRect ret = new HyperRect(new ArrayList<>(points));
        Interval i2 = ret.points.get(desc);
        if (i2.lower > d)
            i2.lower = d;
        return ret;
    }

    public static class Interval {
        private double lower, higher;

        public Interval(double lower, double higher) {
            this.lower = lower;
            this.higher = higher;
        }

        public boolean contains(double point) {
            return lower <= point || point <= higher;

        }

        public void enlarge(double p) {
            if (lower > p)
                lower = p;
            else if (higher < p)
                higher = p;
        }

    }

}
