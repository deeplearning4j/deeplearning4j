package org.deeplearning4j.clustering.vptree;

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
        return buildTreeNode(new ArrayList<T>(points));
    }

    /** List must not be modified after node creation! */
    private static <T extends VpTreePoint<T>> VpTreeNode<T> buildTreeNode(List<T> points) {
        VpTreeNode<T> node = new VpTreeNode<T>(points);

        if (points.size() < MAX_LEAF_SIZE) {
            return node;
        }

        T basePoint = chooseNewVantagePoint(points);
        double distances[] = new double[points.size()];
        double sortedDistances[] = new double[points.size()];

        for (int i = 0; i < points.size(); ++i) {
            distances[i] = basePoint.distance(points.get(i));
            sortedDistances[i] = distances[i];
        }

        Arrays.sort(sortedDistances);
        final double medianDistance = sortedDistances[sortedDistances.length / 2];
        ArrayList<T> leftPoints = new ArrayList<T>(sortedDistances.length);
        ArrayList<T> rightPoints = new ArrayList<T>(sortedDistances.length);

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
        ArrayList<T> candidates = new ArrayList<T>(VANTAGE_POINT_CANDIDATES);
        ArrayList<T> testPoints = new ArrayList<T>(TEST_POINT_COUNT);

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
