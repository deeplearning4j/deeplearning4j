package org.deeplearning4j.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Clusters strings based on fingerprint: for example
 * Two words and TWO words or WORDS TWO would be put together
 * @author Adam Gibson
 *
 */
public class StringCluster extends HashMap<String,Map<String,Integer>> {

    /**
     *
     */
    private static final long serialVersionUID = -4120559428585520276L;
    private FingerPrintKeyer keyer = new FingerPrintKeyer();

    public StringCluster(List<String> list) {
        for(int i = 0; i< list.size(); i++) {
            String s = list.get(i);
            String key = keyer.key(s);
            if (containsKey(key)) {
                Map<String,Integer> m = get(key);
                if (m.containsKey(s)) {
                    m.put(s, m.get(s) + 1);
                } else {
                    m.put(s,1);
                }
            } else {
                Map<String,Integer> m = new TreeMap<String,Integer>();
                m.put(s,1);
                put(key, m);
            }
        }
    }

    public List<Map<String,Integer>> getClusters() {
        List<Map<String,Integer>>_clusters = new ArrayList<Map<String,Integer>>(values());
        Collections.sort(_clusters,new StringCluster.SizeComparator());
        return _clusters;
    }


    public void sort() {
        Collections.sort(new ArrayList<Map<String,Integer>>(values()),new SizeComparator());
    }


    public static class SizeComparator implements Comparator<Map<String,Integer>>, Serializable {
        private static final long serialVersionUID = -1390696157208674054L;
        @Override
        public int compare(Map<String,Integer> o1, Map<String,Integer> o2) {
            int s1 = o1.size();
            int s2 = o2.size();
            if (o1 == o2) {
                int total1 = 0;
                for (int i : o1.values()) {
                    total1 += i;
                }
                int total2 = 0;
                for (int i : o2.values()) {
                    total2 += i;
                }
                return total2 - total1;
            } else {
                return s2 - s1;
            }
        }
    }

}