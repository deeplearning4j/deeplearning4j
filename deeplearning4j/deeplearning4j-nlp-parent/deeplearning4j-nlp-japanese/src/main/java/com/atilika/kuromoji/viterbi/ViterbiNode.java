/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package com.atilika.kuromoji.viterbi;

import com.atilika.kuromoji.dict.Dictionary;

public class ViterbiNode {

    public enum Type {
        KNOWN, UNKNOWN, USER, INSERTED
    }

    private final int wordId;
    private final String surface;
    private final int leftId;
    private final int rightId;

    /**
     * word cost for this node
     */
    private final int wordCost;

    /**
     * minimum path cost found thus far
     */
    private int pathCost;
    private ViterbiNode leftNode;
    private final Type type;
    private final int startIndex;

    public ViterbiNode(int wordId, String surface, int leftId, int rightId, int wordCost, int startIndex, Type type) {
        this.wordId = wordId;
        this.surface = surface;
        this.leftId = leftId;
        this.rightId = rightId;
        this.wordCost = wordCost;
        this.startIndex = startIndex;
        this.type = type;
    }

    public ViterbiNode(int wordId, String word, Dictionary dictionary, int startIndex, Type type) {
        this(wordId, word, dictionary.getLeftId(wordId), dictionary.getRightId(wordId), dictionary.getWordCost(wordId),
                        startIndex, type);
    }

    /**
     * @return the wordId
     */
    public int getWordId() {
        return wordId;
    }

    /**
     * @return the surface
     */
    public String getSurface() {
        return surface;
    }

    /**
     * @return the leftId
     */
    public int getLeftId() {
        return leftId;
    }

    /**
     * @return the rightId
     */
    public int getRightId() {
        return rightId;
    }

    /**
     * @return the cost
     */
    public int getWordCost() {
        return wordCost;
    }

    /**
     * @return the cost
     */
    public int getPathCost() {
        return pathCost;
    }

    /**
     * param cost minimum path cost found this far
     *
     * @param pathCost  cost to set for this node
     */
    public void setPathCost(int pathCost) {
        this.pathCost = pathCost;
    }

    public void setLeftNode(ViterbiNode node) {
        leftNode = node;
    }

    public ViterbiNode getLeftNode() {
        return leftNode;
    }

    public int getStartIndex() {
        return startIndex;
    }

    public Type getType() {
        return type;
    }
}
