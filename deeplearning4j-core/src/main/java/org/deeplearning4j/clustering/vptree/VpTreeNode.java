package org.deeplearning4j.clustering.vptree;

import org.deeplearning4j.berkeley.Counter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Anatoly Borisov
 */
public final class VpTreeNode<T extends VpTreePoint<T>> {

    // The following condition must held:
    // MAX_LEAF_SIZE >= VANTAGE_POINT_CANDIDATES + TEST_POINT_COUNT
    private static final int MAX_LEAF_SIZE = 25;
    private static final int VANTAGE_POINT_CANDIDATES = 5;
    private static final int TEST_POINT_COUNT = 15;

    private VpTreeNode<T> left = null;
    private VpTreeNode<T> right = null;
    private T vantagePoint = null;
    private double leftRadius = 0;
    private final List<T> points;

    public VpTreeNode(List<T> points) {
        this.points = points;
    }



    /**
     * Find top k nearest neighbors near point.
     * @param point the point to get the nearest neighbors
     * @param k the k to choose (if k < 0, an IllegalArgumentException is thrown)
     * @return the k nearest neighbors with their associated distances
     * from the given point
     */
    public Counter<T> findNearByPointsWithDistancesK(T point,int k) {
        if(k <= 0)
            throw new IllegalArgumentException("Illegal k, must be >= 0");
        return findNearbyPoints(new Counter<T>(),point,k);
    }


    private Counter<T> findNearbyPoints(Counter<T> solution,T point, int k) {
        if(solution.size() >= k)
            return solution;
        if (left == null || vantagePoint == null) {
            for (T p : points)
                solution.incrementCount(p,p.distance(point));
            return solution;
        }

        double distanceToLeftCenter = vantagePoint.distance(point);
        if (distanceToLeftCenter  < leftRadius) {
            left.findNearbyPoints(solution,point, k);
        }
        else if (distanceToLeftCenter  >= leftRadius) {
            right.findNearbyPoints(solution,point, k);
        }
        else {
            right.findNearbyPoints(solution,point,k);
            left.findNearbyPoints(solution,point, k);

        }

        solution.keepBottomNKeys(k - 1);
        return solution;

    }



    /**
     * Find top k nearest neighbors near point.
     * @param point the point to get the nearest neighbors
     * @param k the k to choose (if k < 0, an IllegalArgumentException is thrown)
     * @return the k nearest neighbors
     */
    public List<T> findNearByPointsK(T point,int k) {
        if(k <= 0)
            throw new IllegalArgumentException("Illegal k, must be >= 0");
        return findNearbyPoints(new ArrayList<T>(),point,k);
    }

    private List<T> findNearbyPoints(List<T> solution,T point, int k) {
        if(solution.size() >= k)
            return solution;
        Counter<T> counter = new Counter<>();
        if (left == null) {
            List<T> result = new ArrayList<>();
            for (T p : points) {
                result.add(p);
                counter.incrementCount(p,p.distance(point));

            }

            counter.keepBottomNKeys(k);
            result.addAll(counter.getSortedKeys());
            solution.addAll(result);
        }

        double distanceToLeftCenter = vantagePoint.distance(point);
        if (distanceToLeftCenter  < leftRadius) {
            solution.addAll(left.findNearbyPoints(point, k));
        } else if (distanceToLeftCenter  >= leftRadius) {
            solution.addAll(right.findNearbyPoints(point, k));
        } else {
            List<T> result = right.findNearbyPoints(point, k);
            result.addAll(left.findNearbyPoints(point, k));
            solution.addAll(result);
        }

        Counter<T> c = new Counter<>();
        for(T t : solution) {
            c.incrementCount(t,point.distance(t));
        }

        c.keepBottomNKeys(k);
        solution.clear();
        solution.addAll(c.getSortedKeys());
        solution = solution.subList(0,k);
        return solution;

    }




    public List<T> findNearbyPoints(T point, int k) {
        Counter<T> counter = new Counter<>();
        if (left == null) {
            List<T> result = new ArrayList<>();
            for (T p : points) {
                result.add(p);
                counter.incrementCount(p,p.distance(point));

            }

            counter.keepBottomNKeys(k);
            result.addAll(counter.getSortedKeys());
            return result;
        }

        double distanceToLeftCenter = vantagePoint.distance(point);
        if (distanceToLeftCenter  < leftRadius) {
            return left.findNearbyPoints(point, k);
        } else if (distanceToLeftCenter  >= leftRadius) {
            return right.findNearbyPoints(point, k);
        } else {
            List<T> result = right.findNearbyPoints(point, k);
            result.addAll(left.findNearbyPoints(point, k));
            return result;
        }
    }



    public List<T> findNearbyPoints(T point, double maxDistance) {
        if (left == null) {
            List<T> result = new ArrayList<>();
            for (T p : points) {
                if (point.distance(p) <= maxDistance) {
                    result.add(p);
                }
            }
            return result;
        }

        double distanceToLeftCenter = vantagePoint.distance(point);
        if (distanceToLeftCenter + maxDistance < leftRadius) {
            return left.findNearbyPoints(point, maxDistance);
        } else if (distanceToLeftCenter - maxDistance >= leftRadius) {
            return right.findNearbyPoints(point, maxDistance);
        } else {
            List<T> result = right.findNearbyPoints(point, maxDistance);
            result.addAll(left.findNearbyPoints(point, maxDistance));
            return result;
        }
    }

    public static <T extends VpTreePoint<T>> VpTreeNode<T> buildVpTree(List<T> points) {
        return buildTreeNode(new ArrayList<>(points));
    }

    /** List must not be modified after node creation! */
    private static <T extends VpTreePoint<T>> VpTreeNode<T> buildTreeNode(List<T> points) {
        VpTreeNode<T> node = new VpTreeNode<>(points);

        if (points.size() < MAX_LEAF_SIZE)
            return node;


        T basePoint = chooseNewVantagePoint(points);
        double distances[] = new double[points.size()];
        double sortedDistances[] = new double[points.size()];

        for (int i = 0; i < points.size(); ++i) {
            distances[i] = basePoint.distance(points.get(i));
            sortedDistances[i] = distances[i];
        }

        Arrays.sort(sortedDistances);
        final double medianDistance = sortedDistances[sortedDistances.length / 2];
        List<T> leftPoints = new ArrayList<>(sortedDistances.length);
        List<T> rightPoints = new ArrayList<>(sortedDistances.length);

        for (int i = 0; i < distances.length; ++i) {
            if (distances[i] < medianDistance) {
                leftPoints.add(points.get(i));
            } else {
                rightPoints.add(points.get(i));
            }
        }

        for (int i = 0; i < leftPoints.size(); ++i) {
            points.set(i, leftPoints.get(i));
        }

        for (int i = 0; i < rightPoints.size(); ++i) {
            points.set(i + leftPoints.size(), rightPoints.get(i));
        }

        node.vantagePoint = basePoint;
        node.leftRadius = medianDistance;

        node.left = buildTreeNode(points.subList(0, leftPoints.size()));
        node.right = buildTreeNode(points.subList(leftPoints.size(), points.size()));

        return node;
    }

    /** Trying to choose a new vantage point with highest distance deviation to other nodes. */
    private static <T extends VpTreePoint<T>> T chooseNewVantagePoint(List<T> points) {
        List<T> candidates = new ArrayList<T>(VANTAGE_POINT_CANDIDATES);
        List<T> testPoints = new ArrayList<T>(TEST_POINT_COUNT);

        for (int i = 0; i < VANTAGE_POINT_CANDIDATES; ++i) {
            int basePointIndex = i + (int) (Math.random() * (points.size() - i));
            T candidate = points.get(basePointIndex);
            points.set(basePointIndex, points.get(i));
            points.set(i, candidate);
            candidates.add(candidate);
        }

        for (int i = VANTAGE_POINT_CANDIDATES; i < VANTAGE_POINT_CANDIDATES + TEST_POINT_COUNT; ++i) {
            int testPointIndex = i + (int) (Math.random() * (points.size() - i));
            T testPoint = points.get(testPointIndex);
            points.set(testPointIndex, points.get(i));
            points.set(i, testPoint);
            testPoints.add(testPoint);
        }

        T bestBasePoint = points.get(0);
        double bestBasePointSigma = 0;

        for (T basePoint : candidates) {
            double distances[] = new double[TEST_POINT_COUNT];
            for (int i = 0; i < TEST_POINT_COUNT; ++i) {
                distances[i] = basePoint.distance(testPoints.get(i));
            }
            double sigma = sigmaSquare(distances);
            if (sigma > bestBasePointSigma) {
                bestBasePointSigma = sigma;
                bestBasePoint = basePoint;
            }
        }

        return bestBasePoint;
    }

    private static double sigmaSquare(double[] values) {
        double sum = 0;

        for (double value : values) {
            sum += value;
        }

        double avg = sum / values.length;
        double sigmaSq = 0;

        for (double value : values) {
            double dev = value - avg;
            sigmaSq += dev * dev;
        }

        return sigmaSq;
    }
}
