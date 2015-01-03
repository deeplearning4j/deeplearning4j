package org.deeplearning4j.clustering.vptree;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.berkeley.Counter;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Anatoly Borisov
 */
public class VpTreeNodeTest {

    private static Logger log = LoggerFactory.getLogger(VpTreeNodeTest.class);







    @Test
    public void testTopKDistances() {
        List<VpTreePointINDArray> points = new ArrayList<>();
        for (int i = 0; i < 100; ++i) {
            points.add(new VpTreePointINDArray(create(i,i)));
        }

        VpTreeNode<VpTreePointINDArray> node = VpTreeNode.buildVpTree(points);
        VpTreePointINDArray search = new VpTreePointINDArray(create(55.1, 55.2));

        Counter<VpTreePointINDArray> nearbyPoints = node.findNearByPointsWithDistancesK(search, 2);
        Assert.assertEquals(2,nearbyPoints.size());


    }


    @Test
    public void testTopK() {
        List<VpTreePointINDArray> points = new ArrayList<>();
        for (int i = 0; i < 100; ++i) {
            points.add(new VpTreePointINDArray(create(i,i)));
        }

        VpTreeNode<VpTreePointINDArray> node = VpTreeNode.buildVpTree(points);
        VpTreePointINDArray search = new VpTreePointINDArray(create(55.1, 55.2));

        List<VpTreePointINDArray> nearbyPoints = node.findNearByPointsK(search, 2);
        Assert.assertEquals(2,nearbyPoints.size());

        //Assert.assertTrue(nearbyPoints.contains(new VpTreePointINDArray(create(55, 55))));
        //Assert.assertTrue(nearbyPoints.contains(new VpTreePointINDArray(create(56, 56))));

        //nearbyPoints = node.findNearbyPoints(new VpTreePointINDArray(create(10.1, 10.5)), 0.6);
        //Assert.assertTrue(nearbyPoints.contains(new VpTreePointINDArray(create(10, 10))));
        //Assert.assertEquals(1, nearbyPoints.size());
    }

    @Test
    public void testSimpleINDArray() {
        List<VpTreePointINDArray> points = new ArrayList<>();
        for (int i = 0; i < 100; ++i) {
            points.add(new VpTreePointINDArray(create(i,i)));
        }

        VpTreeNode<VpTreePointINDArray> node = VpTreeNode.buildVpTree(points);
        VpTreePointINDArray search = new VpTreePointINDArray(create(55.1, 55.2));

        List<VpTreePointINDArray> nearbyPoints = node.findNearbyPoints(search, 1.5);

        Assert.assertTrue(nearbyPoints.contains(new VpTreePointINDArray(create(55, 55))));
        Assert.assertTrue(nearbyPoints.contains(new VpTreePointINDArray(create(56, 56))));

        nearbyPoints = node.findNearbyPoints(new VpTreePointINDArray(create(10.1, 10.5)), 0.6);
        Assert.assertTrue(nearbyPoints.contains(new VpTreePointINDArray(create(10, 10))));
        Assert.assertEquals(1, nearbyPoints.size());
    }

    @Test
    public void testINDArray() {
        List<VpTreePointINDArray> points = new ArrayList<>();

        for (int i = 0; i < 5000; ++i) {
            points.add(new VpTreePointINDArray(create(Math.random() * 10, Math.random())));
        }

        for (int i = 0; i < 5000; ++i) {
            points.add(new VpTreePointINDArray(create(5 + Math.random() * 5, Math.random())));
        }

        long start = System.currentTimeMillis();
        System.out.println("Building VP-tree...");
        VpTreeNode<VpTreePointINDArray> node = VpTreeNode.buildVpTree(points);
        System.out.println("VP-tree completed, took " + (System.currentTimeMillis() - start) / 1000. + " s");
        start = System.nanoTime();
        List<VpTreePointINDArray> nearbyPoints = node.findNearbyPoints(new VpTreePointINDArray(create(0.1, 0.1)), 0.001);
        System.out.println("VP-tree search completed, took " + (System.nanoTime() - start) + " ns");
        for (VpTreePointINDArray p : nearbyPoints) {
            System.out.println(p);
        }
    }


    private INDArray create(double first,double second) {
        return Nd4j.create(Nd4j.createBuffer(new double[]{first,second}));
    }

    @Test
    public void testSimple() {
        List<VpTreePoint2D> points = new ArrayList<>();
        for (int i = 0; i < 1000; ++i) {
            points.add(new VpTreePoint2D(i, i));
        }
        VpTreeNode<VpTreePoint2D> node = VpTreeNode.buildVpTree(points);
        List<VpTreePoint2D> nearbyPoints = node.findNearbyPoints(new VpTreePoint2D(55.1, 55.2), 1.5);

        Assert.assertTrue(nearbyPoints.contains(new VpTreePoint2D(55, 55)));
        Assert.assertTrue(nearbyPoints.contains(new VpTreePoint2D(56, 56)));

        nearbyPoints = node.findNearbyPoints(new VpTreePoint2D(10.1, 10.5), 0.6);
        Assert.assertTrue(nearbyPoints.contains(new VpTreePoint2D(10, 10)));
        Assert.assertEquals(1, nearbyPoints.size());
    }

    @Test
    public void test() {
        List<VpTreePoint2D> points = new ArrayList<>();

        for (int i = 0; i < 5000; ++i) {
            points.add(new VpTreePoint2D(Math.random() * 10, Math.random()));
        }

        for (int i = 0; i < 5000; ++i) {
            points.add(new VpTreePoint2D(5 + Math.random() * 5, Math.random()));
        }

        long start = System.currentTimeMillis();
        System.out.println("Building VP-tree...");
        VpTreeNode<VpTreePoint2D> node = VpTreeNode.buildVpTree(points);
        System.out.println("VP-tree completed, took " + (System.currentTimeMillis() - start) / 1000. + " s");
        start = System.nanoTime();
        List<VpTreePoint2D> nearbyPoints = node.findNearbyPoints(new VpTreePoint2D(0.1, 0.1), 0.001);
        System.out.println("VP-tree search completed, took " + (System.nanoTime() - start) + " ns");
        for (VpTreePoint2D p : nearbyPoints) {
            System.out.println(p);
        }
    }
}
