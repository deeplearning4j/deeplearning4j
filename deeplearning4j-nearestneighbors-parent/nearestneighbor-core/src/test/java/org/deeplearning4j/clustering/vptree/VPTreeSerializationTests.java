package org.deeplearning4j.clustering.vptree;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * VPTree java serialization tests
 * @author raver119@gmail.com
 */
@Slf4j
public class VPTreeSerializationTests {

    @Test
    public void testSerialization_1() throws Exception {
        val points = Nd4j.rand(new int[] {10, 15});
        val treeA = new VPTree(points, true, 2);

        try (val bos = new ByteArrayOutputStream()) {
            SerializationUtils.serialize(treeA, bos);

            try (val bis = new ByteArrayInputStream(bos.toByteArray())) {
                VPTree treeB = SerializationUtils.deserialize(bis);

                assertEquals(points, treeA.getItems());
                assertEquals(points, treeB.getItems());

                assertEquals(treeA.getWorkers(), treeB.getWorkers());

                val row = points.getRow(1).dup('c');

                val dpListA = new ArrayList<DataPoint>();
                val dListA = new ArrayList<Double>();

                val dpListB = new ArrayList<DataPoint>();
                val dListB = new ArrayList<Double>();

                treeA.search(row, 3, dpListA, dListA);
                treeB.search(row, 3, dpListB, dListB);

                assertTrue(dpListA.size() != 0);
                assertTrue(dListA.size() != 0);

                assertEquals(dpListA.size(), dpListB.size());
                assertEquals(dListA.size(), dListB.size());

                for (int e = 0; e < dpListA.size(); e++) {
                    val rA = dpListA.get(e).getPoint();
                    val rB = dpListB.get(e).getPoint();

                    assertEquals(rA, rB);
                }
            }
        }
    }


    @Test
    public void testNewConstructor_1() {
        val points = Nd4j.rand(new int[] {10, 15});
        val treeA = new VPTree(points, true, 2);

        val rows = Nd4j.tear(points, 1);

        val list = new ArrayList<DataPoint>();

        int idx = 0;
        for (val r: rows)
            list.add(new DataPoint(idx++, r));

        val treeB = new VPTree(list);

        assertEquals(points, treeA.getItems());
        assertEquals(points, treeB.getItems());
    }
}
