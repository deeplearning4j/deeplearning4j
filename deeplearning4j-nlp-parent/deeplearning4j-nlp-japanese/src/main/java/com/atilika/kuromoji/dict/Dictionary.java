/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.atilika.kuromoji.dict;

public interface Dictionary {

    /**
     * Gets the left id of the specified word
     *
     * @param wordId  word id to get left id cost for
     * @return left id cost
     */
    public int getLeftId(int wordId);

    /**
     * Gets the right id of the specified word
     *
     * @param wordId  word id to get right id cost for
     * @return right id cost
     */
    public int getRightId(int wordId);

    /**
     * Gets the word cost of the specified word
     *
     * @param wordId   word id to get word cost for
     * @return word cost
     */
    public int getWordCost(int wordId);

    /**
     * Gets all features of the specified word id
     *
     * @param wordId  word id to get features for
     * @return  All features as a string
     */
    public String getAllFeatures(int wordId);

    /**
     * Gets all features of the specified word id as a String array
     *
     * @param wordId  word id to get features for
     * @return Array with all features
     */
    public String[] getAllFeaturesArray(int wordId);

    /**
     * Gets one or more specific features of a token
     * <p>
     * This is an expert API
     *
     * @param wordId  word id to get features for
     * @param fields array of feature ids. If this array is empty, all features are returned
     * @return Array with specified features
     */
    public String getFeature(int wordId, int... fields);
}
