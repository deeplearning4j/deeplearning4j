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

import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.KnnMinDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 12/29/14.
 */
public class HyperRect implements Serializable {

    //private List<Interval> points;
    private float[] lowerEnds;
    private float[] higherEnds;
    private INDArray lowerEndsIND;
    private INDArray higherEndsIND;

    public HyperRect(float[] lowerEndsIn, float[] higherEndsIn) {
        this.lowerEnds = new float[lowerEndsIn.length];
        this.higherEnds = new float[lowerEndsIn.length];
        System.arraycopy(lowerEndsIn, 0 , this.lowerEnds, 0, lowerEndsIn.length);
        System.arraycopy(higherEndsIn, 0 , this.higherEnds, 0, higherEndsIn.length);
        lowerEndsIND = Nd4j.createFromArray(lowerEnds);
        higherEndsIND = Nd4j.createFromArray(higherEnds);
    }

    public HyperRect(float[] point) {
        this(point, point);
    }

    public HyperRect(Pair<float[], float[]> ends) {
        this(ends.getFirst(), ends.getSecond());
    }


    public void enlargeTo(INDArray point) {
        float[] pointAsArray = point.toFloatVector();
        for (int i = 0; i < lowerEnds.length; i++) {
            float p = pointAsArray[i];
            if (lowerEnds[i] > p)
                lowerEnds[i] = p;
            else if (higherEnds[i] < p)
                higherEnds[i] = p;
        }
    }

    public static Pair<float[],float[]> point(INDArray vector) {
        Pair<float[],float[]> ret = new Pair<>();
        float[] curr = new float[(int)vector.length()];
        for (int i = 0; i < vector.length(); i++) {
            curr[i] = vector.getFloat(i);
        }
        ret.setFirst(curr);
        ret.setSecond(curr);
        return ret;
    }


    /*public List<Boolean> contains(INDArray hPoint) {
        List<Boolean> ret = new ArrayList<>();
        for (int i = 0; i < hPoint.length(); i++) {
            ret.add(lowerEnds[i] <= hPoint.getDouble(i) &&
                    higherEnds[i] >= hPoint.getDouble(i));
        }
        return ret;
    }*/

    public double minDistance(INDArray hPoint, INDArray output) {
        Nd4j.exec(new KnnMinDistance(hPoint, lowerEndsIND, higherEndsIND, output));
        return output.getFloat(0);

        /*double ret = 0.0;
        double[] pointAsArray = hPoint.toDoubleVector();
        for (int i = 0; i < pointAsArray.length; i++) {
           double p = pointAsArray[i];
           if (!(lowerEnds[i] <= p || higherEnds[i] <= p)) {
              if (p < lowerEnds[i])
                 ret += Math.pow((p - lowerEnds[i]), 2);
              else
                 ret += Math.pow((p - higherEnds[i]), 2);
           }
        }
        ret = Math.pow(ret, 0.5);
        return ret;*/
    }

    public HyperRect getUpper(INDArray hPoint, int desc) {
        //Interval interval = points.get(desc);
        float higher = higherEnds[desc];
        float d = hPoint.getFloat(desc);
        if (higher < d)
            return null;
        HyperRect ret = new HyperRect(lowerEnds,higherEnds);
        if (ret.lowerEnds[desc] < d)
            ret.lowerEnds[desc] = d;
        return ret;
    }

    public HyperRect getLower(INDArray hPoint, int desc) {
        //Interval interval = points.get(desc);
        float lower = lowerEnds[desc];
        float d = hPoint.getFloat(desc);
        if (lower > d)
            return null;
        HyperRect ret = new HyperRect(lowerEnds,higherEnds);
        //Interval i2 = ret.points.get(desc);
        if (ret.higherEnds[desc] > d)
            ret.higherEnds[desc] = d;
        return ret;
    }

    @Override
    public String toString() {
        String retVal = "";
        retVal +=  "[";
        for (int i = 0; i < lowerEnds.length; ++i) {
            retVal +=  "("  + lowerEnds[i] + " - " + higherEnds[i] + ") ";
        }
        retVal +=  "]";
        return retVal;
    }
}
