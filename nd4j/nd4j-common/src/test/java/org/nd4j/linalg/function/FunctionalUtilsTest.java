package org.nd4j.linalg.function;

import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class FunctionalUtilsTest {


    @Test
    public void testCoGroup() {
        List<Pair<String,String>> leftMap = new ArrayList<>();
        List<Pair<String,String>> rightMap = new ArrayList<>();

        leftMap.add(Pair.of("cat","adam"));
        leftMap.add(Pair.of("dog","adam"));

        rightMap.add(Pair.of("fish","alex"));
        rightMap.add(Pair.of("cat","alice"));
        rightMap.add(Pair.of("dog","steve"));

        //[(fish,([],[alex])), (dog,([adam],[steve])), (cat,([adam],[alice]))]
        Map<String,Pair<List<String>,List<String>>> assertion = new HashMap<>();
        assertion.put("cat",Pair.of(Arrays.asList("adam"),Arrays.asList("alice")));
        assertion.put("dog",Pair.of(Arrays.asList("adam"),Arrays.asList("steve")));
        assertion.put("fish",Pair.of(Collections.<String>emptyList(),Arrays.asList("alex")));

        Map<String, Pair<List<String>, List<String>>> cogroup = FunctionalUtils.cogroup(leftMap, rightMap);
        assertEquals(assertion,cogroup);

    }

    @Test
    public void testGroupBy() {
        List<Pair<Integer,Integer>> list = new ArrayList<>();
        for(int i = 0; i < 10; i++) {
            for(int j = 0; j < 5; j++) {
                list.add(Pair.of(i, j));
            }
        }

        Map<Integer, List<Integer>> integerIterableMap = FunctionalUtils.groupByKey(list);
        assertEquals(10,integerIterableMap.keySet().size());
        assertEquals(5,integerIterableMap.get(0).size());
    }

    @Test
    public void testMapToPair() {
        Map<String,String> map = new HashMap<>();
        for(int i = 0; i < 5; i++) {
            map.put(String.valueOf(i),String.valueOf(i));
        }

        List<Pair<String, String>> pairs = FunctionalUtils.mapToPair(map);
        assertEquals(map.size(),pairs.size());
    }

}
