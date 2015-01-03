package org.deeplearning4j.clustering.vptree;

/**
 * @author Anatoly Borisov
 */
public class VpTreePoint2D implements VpTreePoint<VpTreePoint2D> {
    public final double x;
    public final double y;

    public VpTreePoint2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public double distance(VpTreePoint2D p) {
        double dx = x - p.x;
        double dy = y - p.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public String toString() {
        return "(" + x + ", " + y + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        VpTreePoint2D that = (VpTreePoint2D) o;

        if (Double.compare(that.x, x) != 0) return false;
        if (Double.compare(that.y, y) != 0) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        temp = Double.doubleToLongBits(x);
        result = (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(y);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}
