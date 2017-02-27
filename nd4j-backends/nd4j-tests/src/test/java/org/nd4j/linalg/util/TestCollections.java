package org.nd4j.linalg.util;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.collection.CompactHeapStringList;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestCollections extends BaseNd4jTest {

    public TestCollections(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testCompactHeapStringList() {

        int[] reallocSizeBytes = new int[] {1024, 1048576};
        int[] intReallocSizeBytes = new int[] {1024, 1048576};

        int numElementsToTest = 10000;
        int minLength = 1;
        int maxLength = 1048;

        Random r = new Random(12345);

        List<String> compare = new ArrayList<>(numElementsToTest);
        for (int i = 0; i < numElementsToTest; i++) {
            int thisLength = minLength + r.nextInt(maxLength);
            char[] c = new char[thisLength];
            for (int j = 0; j < c.length; j++) {
                c[j] = (char) r.nextInt(65536);
            }
            String s = new String(c);
            compare.add(s);
        }


        for (int rb : reallocSizeBytes) {
            for (int irb : intReallocSizeBytes) {
                //                System.out.println(rb + "\t" + irb);
                List<String> list = new CompactHeapStringList(rb, irb);

                assertTrue(list.isEmpty());
                assertEquals(0, list.size());


                for (int i = 0; i < numElementsToTest; i++) {
                    String s = compare.get(i);
                    list.add(s);

                    assertEquals(i + 1, list.size());
                    String s2 = list.get(i);
                    assertEquals(s, s2);
                }

                assertEquals(numElementsToTest, list.size());

                assertEquals(list, compare);
                assertEquals(compare, list);
                assertEquals(compare, Arrays.asList(list.toArray()));

                for (int i = 0; i < numElementsToTest; i++) {
                    assertEquals(i, list.indexOf(compare.get(i)));
                }

                Iterator<String> iter = list.iterator();
                int count = 0;
                while (iter.hasNext()) {
                    String s = iter.next();
                    assertEquals(s, compare.get(count++));
                }
                assertEquals(numElementsToTest, count);
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
