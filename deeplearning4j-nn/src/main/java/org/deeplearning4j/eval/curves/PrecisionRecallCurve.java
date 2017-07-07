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

    private Double area;

    public PrecisionRecallCurve(@JsonProperty("threshold") double[] threshold,
                    @JsonProperty("precision") double[] precision, @JsonProperty("recall") double[] recall) {
        this.threshold = threshold;
        this.precision = precision;
        this.recall = recall;
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
     * Get the point (index, threshold, precision, recall) at the given threshold.
     * Note that if the threshold is not found exactly, the next highest threshold exceeding the requested threshold
     * is returned
     *
     * @param threshold Threshold to get the point for
     * @return point (index, threshold, precision, recall) at the given threshold
     */
    public Point getPointAtThreshold(double threshold){

        //Return (closest) point number, precision, recall, whether it's interpolated or not

        //Binary search to find closest threshold

        int idx = Arrays.binarySearch(this.threshold, threshold);
        if (idx < 0) {
            //Not found (usual case)
            /*
            index of the search key, if it is contained in the array;
     *         otherwise, <tt>(-(<i>insertion point</i>) - 1)</tt>.  The
     *         <i>insertion point</i> is defined as the point at which the
     *         key would be inserted into the array: the index of the first
     *         element greater than the key, or <tt>a.length</tt> if all
     *         elements in the array are less than the specified key.
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
    public Point getPointAtPrecision(double precision){
        //Find the LOWEST threshold that gives the specified precision

        for( int i=0; i<this.precision.length; i++ ){
            if(this.precision[i] >= precision){
                return new Point(i, threshold[i], this.precision[i], recall[i]);
            }
        }

        //Not found, return last point. Should never happen though...
        int i = threshold.length-1;
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
    public Point getPointAtRecall(double recall){
        //Find the HIGHEST threshold that gives the specified recall
        for( int i=this.recall.length-1; i>=0; i-- ){
            if(this.recall[i] >= recall){
                return new Point(i, threshold[i], precision[i], this.recall[i]);
            }
        }

        //Not found - return first point. Should never happen...
        return new Point(0, threshold[0], precision[0], this.recall[0]);
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
}
