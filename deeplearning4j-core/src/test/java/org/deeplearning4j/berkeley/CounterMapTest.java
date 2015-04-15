package org.deeplearning4j.berkeley;

import static org.junit.Assert.*;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by agibsoncccc on 4/15/15.
 */
public class CounterMapTest {
    private static Logger log = LoggerFactory.getLogger(CounterMapTest.class);


    @Test
    public void testParallel() {
        List<Integer> items = new ArrayList<>();
        for(int i  = 0; i < 1e6; i++)
            items.add(i);

        CounterMap<Integer,Integer> pairWise = CounterMap.runPairWise(items, new CounterMap.CountFunction<Integer>() {
            @Override
            public double count(Integer v1, Integer v2) {
                return 1.0;
            }
        });

        Iterator<Pair<Integer,Integer>> iter = pairWise.getPairIterator();
        while(iter.hasNext()) {
            Pair<Integer,Integer> pair = iter.next();
            double count1 = pairWise.getCount(pair.getFirst(),pair.getSecond());
            double count2 = pairWise.getCount(pair.getSecond(),pair.getFirst());
            assertEquals(count1,count2,1e-1);
        }

    }

}
