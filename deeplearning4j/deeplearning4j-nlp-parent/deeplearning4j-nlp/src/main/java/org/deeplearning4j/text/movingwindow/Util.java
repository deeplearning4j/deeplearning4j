/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.text.movingwindow;

import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.CounterMap;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;


public class Util {


    private Util() {}

    /**
     * Returns a thread safe counter map
     * @return
     */
    public static <K, V> CounterMap<K, V> parallelCounterMap() {
        CounterMap<K, V> totalWords = new CounterMap<>();
        return totalWords;
    }


    /**
     * Returns a thread safe counter
     * @return
     */
    public static <K> Counter<K> parallelCounter() {
        Counter<K> totalWords = new Counter<>();
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
