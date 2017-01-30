/*
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.deeplearning4j.util;

import java.util.Random;

/**
 * Created by Alex on 24/01/2017.
 */
public class RandomUtils {

    /**
     * Randomly shuffle the specified integer array using a Fisher-Yates shuffle algorithm
     * @param toShuffle Array to shuffle
     * @param random    RNG to use for shuffling
     */
    public static void shuffleInPlace(int[] toShuffle, Random random){
        //Fisher-Yates shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        for( int i=0; i<toShuffle.length-1; i++ ){
            int j = i + random.nextInt(toShuffle.length-i);
            int temp = toShuffle[i];
            toShuffle[i] = toShuffle[j];
            toShuffle[j] = temp;
        }
    }
}
