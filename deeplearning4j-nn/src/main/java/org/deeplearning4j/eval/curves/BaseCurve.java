package org.deeplearning4j.eval.curves;

/**
 * Created by Alex on 17/06/2017.
 */
public abstract class BaseCurve {
    public static final int DEFAULT_FORMAT_PREC = 4;


    public abstract int numPoints();

    public abstract double[] getX();

    public abstract double[] getY();

    public abstract String getTitle();

    protected double calculateArea(){
        return calculateArea(getX(), getY());
    }

    protected double calculateArea(double[] x, double[] y){
        int nPoints = x.length;
        double area = 0.0;
        for (int i = 0; i < nPoints - 1; i++) {
            double xLeft = x[i];
            double yLeft = y[i];
            double xRight = x[i + 1];
            double yRight = y[i + 1];

            //y axis: TPR
            //x axis: FPR
            double deltaX = Math.abs(xRight - xLeft); //Iterating in threshold order, so FPR decreases as threshold increases
            double avg = (yRight + yLeft) / 2.0;

            area += deltaX * avg;
        }

        return area;
    }

    protected String format(double d, int precision){
        return String.format("%." + precision + "f", d);
    }

}
