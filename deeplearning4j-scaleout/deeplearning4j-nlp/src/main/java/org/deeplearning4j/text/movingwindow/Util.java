package org.deeplearning4j.text.movingwindow;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.MapFactory;


public class Util {



    /**
     * Returns a thread safe counter map
     * @return
     */
    public static <K,V> CounterMap<K,V> parallelCounterMap() {
        MapFactory<K,Double> factory = new MapFactory<K,Double>() {

            private static final long serialVersionUID = 5447027920163740307L;

            @Override
            public Map<K, Double> buildMap() {
                return new java.util.concurrent.ConcurrentHashMap<>();
            }

        };

        CounterMap<K,V> totalWords = new CounterMap(factory,factory);
        return totalWords;
    }


    /**
     * Returns a thread safe counter
     * @return
     */
    public static <K> Counter<K> parallelCounter() {
        MapFactory<K,Double> factory = new MapFactory<K,Double>() {

            private static final long serialVersionUID = 5447027920163740307L;

            @Override
            public Map<K, Double> buildMap() {
                return new java.util.concurrent.ConcurrentHashMap<>();
            }

        };

        Counter<K> totalWords = new Counter<>(factory);
        return totalWords;
    }



    public static boolean matchesAnyStopWord(List<String> stopWords,String word) {
        for(String s : stopWords)
            if(s.equalsIgnoreCase(word))
                return true;
        return false;
    }

    public static Level disableLogging() {
        Logger logger = Logger.getLogger("org.apache.uima");
        while (logger.getLevel() == null) {
            logger = logger.getParent();
        }
        Level level = logger.getLevel();
        logger.setLevel(Level.OFF);
        return level;
    }


}
