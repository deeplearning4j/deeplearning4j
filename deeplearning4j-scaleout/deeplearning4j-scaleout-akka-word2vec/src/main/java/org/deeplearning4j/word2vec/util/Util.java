package org.deeplearning4j.word2vec.util;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.MapFactory;


public class Util {

    /**
     * Returns a thread safe counter
     * @return
     */
    public static Counter<String> parallelCounter() {
        MapFactory<String,Double> factory = new MapFactory<String,Double>() {

            private static final long serialVersionUID = 5447027920163740307L;

            @Override
            public Map<String, Double> buildMap() {
                return new java.util.concurrent.ConcurrentHashMap<String,Double>();
            }

        };

        Counter<String> totalWords = new Counter<String>(factory);
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
