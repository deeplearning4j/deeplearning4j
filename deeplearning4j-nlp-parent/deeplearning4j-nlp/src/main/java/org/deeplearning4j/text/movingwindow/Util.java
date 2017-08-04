/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.movingwindow;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.MapFactory;

import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;


public class Util {


    private Util() {}

    /**
     * Returns a thread safe counter map
     * @return
     */
    public static <K, V> CounterMap<K, V> parallelCounterMap() {
        MapFactory<K, Double> factory = new MapFactory<K, Double>() {

            private static final long serialVersionUID = 5447027920163740307L;

            @Override
            public Map<K, Double> buildMap() {
                return new java.util.concurrent.ConcurrentHashMap<>();
            }

        };

        CounterMap<K, V> totalWords = new CounterMap(factory, factory);
        return totalWords;
    }


    /**
     * Returns a thread safe counter
     * @return
     */
    public static <K> Counter<K> parallelCounter() {
        MapFactory<K, Float> factory = new MapFactory<K, Float>() {

            private static final long serialVersionUID = 5447027920163740307L;

            @Override
            public Map<K, Float> buildMap() {
                return new java.util.concurrent.ConcurrentHashMap<>();
            }

        };

        Counter<K> totalWords = new Counter<>(factory);
        return totalWords;
    }



    public static boolean matchesAnyStopWord(List<String> stopWords, String word) {
        for (String s : stopWords)
            if (s.equalsIgnoreCase(word))
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
