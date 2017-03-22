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
package com.atilika.kuromoji.viterbi;

import com.atilika.kuromoji.TokenizerBase;
import com.atilika.kuromoji.dict.ConnectionCosts;
import com.atilika.kuromoji.dict.UnknownDictionary;

import java.util.LinkedList;
import java.util.List;

public class ViterbiSearcher {

    private static final int DEFAULT_COST = Integer.MAX_VALUE;

    private final ConnectionCosts costs;
    private final UnknownDictionary unknownDictionary;

    private int kanjiPenaltyLengthTreshold;
    private int otherPenaltyLengthThreshold;
    private int kanjiPenalty;
    private int otherPenalty;

    private final TokenizerBase.Mode mode;

    public ViterbiSearcher(TokenizerBase.Mode mode, ConnectionCosts costs, UnknownDictionary unknownDictionary,
                    List<Integer> penalties) {
        if (!penalties.isEmpty()) {
            this.kanjiPenaltyLengthTreshold = penalties.get(0);
            this.kanjiPenalty = penalties.get(1);
            this.otherPenaltyLengthThreshold = penalties.get(2);
            this.otherPenalty = penalties.get(3);
        }

        this.mode = mode;
        this.costs = costs;
        this.unknownDictionary = unknownDictionary;
    }

    /**
     * Find best path from input lattice.
     *
     * @param lattice the result of build method
     * @return List of ViterbiNode which consist best path
     */
    public List<ViterbiNode> search(ViterbiLattice lattice) {

        ViterbiNode[][] endIndexArr = calculatePathCosts(lattice);
        LinkedList<ViterbiNode> result = backtrackBestPath(endIndexArr[0][0]);

        return result;
    }

    private ViterbiNode[][] calculatePathCosts(ViterbiLattice lattice) {
        ViterbiNode[][] startIndexArr = lattice.getStartIndexArr();
        ViterbiNode[][] endIndexArr = lattice.getEndIndexArr();

        for (int i = 1; i < startIndexArr.length; i++) {

            if (startIndexArr[i] == null || endIndexArr[i] == null) { // continue since no array which contains ViterbiNodes exists. Or no previous node exists.
                continue;
            }

            for (ViterbiNode node : startIndexArr[i]) {
                if (node == null) { // If array doesn't contain ViterbiNode any more, continue to next index
                    break;
                }

                updateNode(endIndexArr[i], node);
            }
        }
        return endIndexArr;
    }

    private void updateNode(ViterbiNode[] viterbiNodes, ViterbiNode node) {
        int backwardConnectionId = node.getLeftId();
        int wordCost = node.getWordCost();
        int leastPathCost = DEFAULT_COST;

        for (ViterbiNode leftNode : viterbiNodes) {
            // If array doesn't contain any more ViterbiNodes, continue to next index
            if (leftNode == null) {
                return;
            } else {
                // cost = [total cost from BOS to previous node] + [connection cost between previous node and current node] + [word cost]
                int pathCost = leftNode.getPathCost() + costs.get(leftNode.getRightId(), backwardConnectionId)
                                + wordCost;

                // Add extra cost for long nodes in "Search mode".
                if (mode == TokenizerBase.Mode.SEARCH || mode == TokenizerBase.Mode.EXTENDED) {
                    pathCost += getPenaltyCost(node);
                }

                // If total cost is lower than before, set current previous node as best left node (previous means left).
                if (pathCost < leastPathCost) {
                    leastPathCost = pathCost;
                    node.setPathCost(leastPathCost);
                    node.setLeftNode(leftNode);
                }
            }
        }
    }

    private int getPenaltyCost(ViterbiNode node) {
        int pathCost = 0;
        String surface = node.getSurface();
        int length = surface.length();

        if (length > kanjiPenaltyLengthTreshold) {
            if (isKanjiOnly(surface)) { // Process only Kanji keywords
                pathCost += (length - kanjiPenaltyLengthTreshold) * kanjiPenalty;
            } else if (length > otherPenaltyLengthThreshold) {
                pathCost += (length - otherPenaltyLengthThreshold) * otherPenalty;
            }
        }
        return pathCost;
    }

    private boolean isKanjiOnly(String surface) {
        for (int i = 0; i < surface.length(); i++) {
            char c = surface.charAt(i);

            if (Character.UnicodeBlock.of(c) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) {
                return false;
            }
        }
        return true;
    }

    private LinkedList<ViterbiNode> backtrackBestPath(ViterbiNode eos) {
        ViterbiNode node = eos;
        LinkedList<ViterbiNode> result = new LinkedList<>();

        result.add(node);

        while (true) {
            ViterbiNode leftNode = node.getLeftNode();

            if (leftNode == null) {
                break;
            } else {
                // Extended mode converts unknown word into unigram nodes
                if (mode == TokenizerBase.Mode.EXTENDED && leftNode.getType() == ViterbiNode.Type.UNKNOWN) {
                    LinkedList<ViterbiNode> uniGramNodes = convertUnknownWordToUnigramNode(leftNode);
                    result.addAll(uniGramNodes);
                } else {
                    result.addFirst(leftNode);
                }
                node = leftNode;
            }
        }
        return result;
    }

    private LinkedList<ViterbiNode> convertUnknownWordToUnigramNode(ViterbiNode node) {
        LinkedList<ViterbiNode> uniGramNodes = new LinkedList<>();
        int unigramWordId = 0;
        String surface = node.getSurface();

        for (int i = surface.length(); i > 0; i--) {
            String word = surface.substring(i - 1, i);
            int startIndex = node.getStartIndex() + i - 1;

            ViterbiNode uniGramNode = new ViterbiNode(unigramWordId, word, unknownDictionary, startIndex,
                            ViterbiNode.Type.UNKNOWN);
            uniGramNodes.addFirst(uniGramNode);
        }

        return uniGramNodes;
    }
}
