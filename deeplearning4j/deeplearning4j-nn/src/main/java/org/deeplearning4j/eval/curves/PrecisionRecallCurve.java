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

package org.deeplearning4j.eval.curves;

import com.google.common.base.Preconditions;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * Precision recall curve: A set of (recall, precision) points and different thresholds
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"area"}, callSuper = false)
public class PrecisionRecallCurve extends BaseCurve {

    private double[] threshold;
    private double[] precision;
    private double[] recall;
    private int[] tpCount;
    private int[] fpCount;
    private int[] fnCount;
    private int totalCount;

    private Double area;

    public PrecisionRecallCurve(@JsonProperty("threshold") double[] threshold,
                    @JsonProperty("precision") double[] precision, @JsonProperty("recall") double[] recall,
                    @JsonProperty("tpCount") int[] tpCount, @JsonProperty("fpCount") int[] fpCount,
                    @JsonProperty("fnCount") int[] fnCount, @JsonProperty("totalCount") int totalCount) {
        this.threshold = threshold;
        this.precision = precision;
        this.recall = recall;
        this.tpCount = tpCount;
        this.fpCount = fpCount;
        this.fnCount = fnCount;
        this.totalCount = totalCount;
    }

    @Override
    public int numPoints() {
        return threshold.length;
    }

    @Override
    public double[] getX() {
        return recall;
    }

    @Override
    public double[] getY() {
        return precision;
    }

    @Override
    public String getTitle() {
        return "Precision-Recall Curve (Area=" + format(calculateAUPRC(), DEFAULT_FORMAT_PREC) + ")";
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Threshold of a given point
     */
    public double getThreshold(int i) {
        Preconditions.checkArgument(i >= 0 && i < threshold.length, "Invalid index: " + i);
        return threshold[i];
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Precision of a given point
     */
    public double getPrecision(int i) {
        Preconditions.checkArgument(i >= 0 && i < precision.length, "Invalid index: " + i);
        return precision[i];
    }

    /**
     * @param i Point number, 0 to numPoints()-1 inclusive
     * @return Recall of a given point
     */
    public double getRecall(int i) {
        Preconditions.checkArgument(i >= 0 && i < recall.length, "Invalid index: " + i);
        return recall[i];
    }

    /**
     * @return The area under the precision recall curve
     */
    public double calculateAUPRC() {
        if (area != null) {
            return area;
        }

        area = calculateArea();
        return area;
    }

    /**
     * Get the point (index, threshold, precision, recall) at the given threshold.<br>
     * Note that if the threshold is not found exactly, the next highest threshold exceeding the requested threshold
     * is returned
     *
     * @param threshold Threshold to get the point for
     * @return point (index, threshold, precision, recall) at the given threshold
     */
    public Point getPointAtThreshold(double threshold) {

        //Return (closest) point number, precision, recall, whether it's interpolated or not

        //Binary search to find closest threshold

        int idx = Arrays.binarySearch(this.threshold, threshold);
        if (idx < 0) {
            //Not found (usual case). binarySearch javadoc:
            /*
            index of the search key, if it is contained in the array;
            otherwise, (-(insertion point) - 1).  The
            insertion point is defined as the point at which the
            key would be inserted into the array: the index of the first
            element greater than the key, or a.length if all
            elements in the array are less than the specified key.
            */
            idx = -idx - 1;
        }

        //At this point: idx = exact, on the next highest
        double thr = this.threshold[idx];
        double pr = precision[idx];
        double rec = recall[idx];

        return new Point(idx, thr, pr, rec);
    }

    /**
     * Get the point (index, threshold, precision, recall) at the given precision.<br>
     * Specifically, return the points at the lowest threshold that has precision equal to or greater than the
     * requested precision.
     *
     * @param precision Precision to get the point for
     * @return point (index, threshold, precision, recall) at (or closest exceeding) the given precision
     */
    public Point getPointAtPrecision(double precision) {
        //Find the LOWEST threshold that gives the specified precision

        for (int i = 0; i < this.precision.length; i++) {
            if (this.precision[i] >= precision) {
                return new Point(i, threshold[i], this.precision[i], recall[i]);
            }
        }

        //Not found, return last point. Should never happen though...
        int i = threshold.length - 1;
        return new Point(i, threshold[i], this.precision[i], this.recall[i]);
    }

    /**
     * Get the point (index, threshold, precision, recall) at the given recall.<br>
     * Specifically, return the points at the highest threshold that has recall equal to or greater than the
     * requested recall.
     *
     * @param recall Recall to get the point for
     * @return point (index, threshold, precision, recall) at (or closest exceeding) the given recall
     */
    public Point getPointAtRecall(double recall) {
        Point foundPoint = null;
        //Find the HIGHEST threshold that gives the specified recall
        for (int i = this.recall.length - 1; i >= 0; i--) {
                if (this.recall[i] >= recall) {
                        if (foundPoint == null ||(this.recall[i] == foundPoint.getRecall() && this.precision[i] >= foundPoint.getPrecision())) {
                                foundPoint = new Point(i, threshold[i], precision[i], this.recall[i]);
                        }
                }
        }
        if (foundPoint == null){
        	//Not found - return first point. Should never happen...
        	foundPoint = new Point(0, threshold[0], precision[0], this.recall[0]);
        }
        return foundPoint;
    }

    /**
     * Get the binary confusion matrix for the given threshold. As per {@link #getPointAtThreshold(double)},
     * if the threshold is not found exactly, the next highest threshold exceeding the requested threshold
     * is returned
     *
     * @param threshold Threshold at which to get the confusion matrix
     * @return Binary confusion matrix
     */
    public Confusion getConfusionMatrixAtThreshold(double threshold) {
        Point p = getPointAtThreshold(threshold);
        int idx = p.idx;
        int tn = totalCount - (tpCount[idx] + fpCount[idx] + fnCount[idx]);
        return new Confusion(p, tpCount[idx], fpCount[idx], fnCount[idx], tn);
    }

    /**
     * Get the binary confusion matrix for the given position. As per {@link #getPointAtThreshold(double)}.
     *
     * @param point Position at which to get the binary confusion matrix
     * @return Binary confusion matrix
     */
    public Confusion getConfusionMatrixAtPoint(int point) {
        return getConfusionMatrixAtThreshold(threshold[point]);
    }

    public static PrecisionRecallCurve fromJson(String json) {
        return fromJson(json, PrecisionRecallCurve.class);
    }

    public static PrecisionRecallCurve fromYaml(String yaml) {
        return fromYaml(yaml, PrecisionRecallCurve.class);
    }

    @AllArgsConstructor
    @Data
    public static class Point {
        private final int idx;
        private final double threshold;
        private final double precision;
        private final double recall;
    }

    @AllArgsConstructor
    @Data
    public static class Confusion {
        private final Point point;
        private final int tpCount;
        private final int fpCount;
        private final int fnCount;
        private final int tnCount;
    }
}
