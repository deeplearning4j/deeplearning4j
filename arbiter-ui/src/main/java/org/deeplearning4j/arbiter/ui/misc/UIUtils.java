package org.deeplearning4j.arbiter.ui.misc;

import org.joda.time.Period;
import org.joda.time.PeriodType;
import org.joda.time.format.PeriodFormatter;
import org.joda.time.format.PeriodFormatterBuilder;

/**
 * Created by Alex on 20/07/2017.
 */
public class UIUtils {

    /**
     * Convert the "messy" min/max values on a dataset to something clean. For example, 0.895732 becomes 1.0
     *
     * @param max   Maximum data point value
     * @param min   Minimum data point value
     * @param nTick Number of tick marks desired on chart (good setting: 5)
     * @return double[] of length 2 - with new minimum and maximum
     */
    public static double[] graphNiceRange(double max, double min, int nTick){
        if(max == min || !Double.isFinite(max)){
            if(max == 0.0 || !Double.isFinite(max)){
                return new double[]{0.0, 1.0};
            }

            return graphNiceRange(1.5 * max, 0.5 * max, nTick);
        }

        double range = niceNum(max-min, false);
        double d = niceNum(range / (nTick-1), true );
        double graphMin = Math.floor(min/d)*d;
        double graphMax = Math.ceil(max/d)*d;


        return new double[]{graphMin, graphMax};
    }

    public static double niceNum(double x, boolean round){
        double exp = Math.floor(Math.log10(x));
        double f = x / Math.pow(10, exp);

        double nf;
        if(round){
            if(f < 1.5 ){
                nf = 1;
            } else if( f < 3){
                nf = 2;
            } else if( f < 7){
                nf = 5;
            } else {
                nf = 10;
            }
        } else {
            if(f <= 1 ){
                nf = 1;
            } else if( f <= 2){
                nf = 2;
            } else if( f <= 5){
                nf = 5;
            } else {
                nf = 10;
            }
        }
        return nf * Math.pow(10, exp);
    }

    /**
     * Format the duration in milliseconds to a human readable String, with "yr", "days", "hr" etc prefixes
     *
     *
     * @param durationMs Duration in milliseconds
     * @return Human readable string
     */
    public static String formatDuration(long durationMs){
        Period period = Period.seconds((int)(durationMs/1000L));
        Period p2 = period.normalizedStandard(PeriodType.yearMonthDayTime());

        PeriodFormatter formatter = new PeriodFormatterBuilder()
                .appendYears()
                .appendSuffix(" yr ")
                .appendMonths()
                .appendSuffix(" months ")
                .appendDays()
                .appendSuffix(" days ")
                .appendHours()
                .appendSuffix(" hr ")
                .appendMinutes()
                .appendSuffix(" min ")
                .appendSeconds()
                .appendSuffix(" sec")
                .toFormatter();

        return formatter.print(p2);
    }
}
