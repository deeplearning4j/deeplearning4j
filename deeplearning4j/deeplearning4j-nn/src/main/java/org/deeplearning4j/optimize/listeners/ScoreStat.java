/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.optimize.listeners;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Score statistics container used by {@link CollectScoresIterationListener}.
 */
public class ScoreStat {
    public static final int BUCKET_LENGTH = 10000;

    private int position = 0;
    private int bucketNumber = 1;
    private List<long[]> indexes;
    private List<double[]> scores;

    public ScoreStat() {
        indexes = new ArrayList<>(1);
        indexes.add(new long[BUCKET_LENGTH]);
        scores = new ArrayList<>(1);
        scores.add(new double[BUCKET_LENGTH]);
    }

    public List<long[]> getIndexes() {
        return indexes;
    }

    public List<double[]> getScores() {
        return scores;
    }

    public int getPosition() {
        return position;
    }

    public long[] getEffectiveIndexes() {
        return Arrays.copyOfRange(indexes.get(0), 0, position);
    }

    public double[] getEffectiveScores() {
        return Arrays.copyOfRange(scores.get(0), 0, position);
    }


    /*
        Originally scores array is initialized with BUCKET_LENGTH size.
        When data doesn't fit there - arrays size is increased for BUCKET_LENGTH,
        old data is copied and bucketNumber (counter of reallocations) being incremented.

        If we got more score points than MAX_VALUE - they are put to another item of scores list.
     */
    void reallocateGuard() {
        if (position >= BUCKET_LENGTH * bucketNumber) {

            long fullLength = (long)BUCKET_LENGTH * bucketNumber;

            if (position == Integer.MAX_VALUE || fullLength >= Integer.MAX_VALUE) {
                position = 0;
                long[] newIndexes = new long[BUCKET_LENGTH];
                double[] newScores = new double[BUCKET_LENGTH];
                indexes.add(newIndexes);
                scores.add(newScores);
            }
            else {
                long[] newIndexes = new long[(int)fullLength + BUCKET_LENGTH];
                double[] newScores = new double[(int)fullLength + BUCKET_LENGTH];
                System.arraycopy(indexes.get(indexes.size()-1), 0, newIndexes, 0, (int)fullLength);
                System.arraycopy(scores.get(scores.size()-1), 0, newScores, 0, (int)fullLength);
                scores.remove(scores.size()-1);
                indexes.remove(indexes.size()-1);
                int lastIndex = scores.size() == 0 ? 0 : scores.size()-1;
                scores.add(lastIndex, newScores);
                indexes.add(lastIndex, newIndexes);
            }
            bucketNumber += 1;
        }
    }

    public void addScore(long index, double score) {
        reallocateGuard();
        scores.get(scores.size() - 1)[position] = score;
        indexes.get(indexes.size() - 1)[position] = index;
        position += 1;
    }
}
