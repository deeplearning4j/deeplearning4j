package org.deeplearning4j.datasets.iterator;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.datasets.iterator.tools.DataSetGenerator;
import org.junit.Test;

import static org.junit.Assert.*;

@Slf4j
public class JointMultiDataSetIteratorTests {

    @Test (timeout = 20000L)
    public void testJMDSI_1() {
        val iter0 = new DataSetGenerator(32, new int[]{3, 3}, new int[]{2, 2});
        val iter1 = new DataSetGenerator(32, new int[]{3, 3, 3}, new int[]{2, 2, 2});
        val iter2 = new DataSetGenerator(32, new int[]{3, 3, 3, 3}, new int[]{2, 2, 2, 2});

        val mdsi = new JointMultiDataSetIterator(iter0, iter1, iter2);

        int cnt = 0;
        while (mdsi.hasNext()) {
            val mds = mdsi.next();

            assertNotNull(mds);

            val f = mds.getFeatures();
            val l = mds.getLabels();

            val fm = mds.getFeaturesMaskArrays();
            val lm = mds.getLabelsMaskArrays();

            assertNotNull(f);
            assertNotNull(l);
            assertNull(fm);
            assertNull(lm);

            assertArrayEquals(new long[]{3, 3}, f[0].shape());
            assertArrayEquals(new long[]{3, 3, 3}, f[1].shape());
            assertArrayEquals(new long[]{3, 3, 3, 3}, f[2].shape());

            assertEquals(cnt, (int) f[0].getDouble(0));
            assertEquals(cnt, (int) f[1].getDouble(0));
            assertEquals(cnt, (int) f[2].getDouble(0));

            assertArrayEquals(new long[]{2, 2}, l[0].shape());
            assertArrayEquals(new long[]{2, 2, 2}, l[1].shape());
            assertArrayEquals(new long[]{2, 2, 2, 2}, l[2].shape());

            cnt++;
        }

        assertEquals(32, cnt);
    }


    @Test (timeout = 20000L)
    public void testJMDSI_2() {
        val iter0 = new DataSetGenerator(32, new int[]{3, 3}, new int[]{2, 2});
        val iter1 = new DataSetGenerator(32, new int[]{3, 3, 3}, new int[]{2, 2, 2});
        val iter2 = new DataSetGenerator(32, new int[]{3, 3, 3, 3}, new int[]{2, 2, 2, 2});

        val mdsi = new JointMultiDataSetIterator(1, iter0, iter1, iter2);

        int cnt = 0;
        while (mdsi.hasNext()) {
            val mds = mdsi.next();

            assertNotNull(mds);

            val f = mds.getFeatures();
            val l = mds.getLabels();

            val fm = mds.getFeaturesMaskArrays();
            val lm = mds.getLabelsMaskArrays();

            assertNotNull(f);
            assertNotNull(l);
            assertNull(fm);
            assertNull(lm);

            assertArrayEquals(new long[]{3, 3}, f[0].shape());
            assertArrayEquals(new long[]{3, 3, 3}, f[1].shape());
            assertArrayEquals(new long[]{3, 3, 3, 3}, f[2].shape());

            assertEquals(cnt, (int) f[0].getDouble(0));
            assertEquals(cnt, (int) f[1].getDouble(0));
            assertEquals(cnt, (int) f[2].getDouble(0));

            assertEquals(1, l.length);

            assertArrayEquals(new long[]{2, 2, 2}, l[0].shape());
            assertEquals(cnt, (int) l[0].getDouble(0));

            cnt++;
        }

        assertEquals(32, cnt);
    }
}
