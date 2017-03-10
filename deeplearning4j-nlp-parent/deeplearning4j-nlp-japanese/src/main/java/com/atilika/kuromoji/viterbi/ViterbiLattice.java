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

public class ViterbiLattice {

    private static final String BOS = "BOS";
    private static final String EOS = "EOS";

    private final int dimension;
    private final ViterbiNode[][] startIndexArr;
    private final ViterbiNode[][] endIndexArr;
    private final int[] startSizeArr;
    private final int[] endSizeArr;

    public ViterbiLattice(int dimension) {
        this.dimension = dimension;
        startIndexArr = new ViterbiNode[dimension][];
        endIndexArr = new ViterbiNode[dimension][];
        startSizeArr = new int[dimension];
        endSizeArr = new int[dimension];
    }

    public ViterbiNode[][] getStartIndexArr() {
        return startIndexArr;
    }

    public ViterbiNode[][] getEndIndexArr() {
        return endIndexArr;
    }

    public int[] getStartSizeArr() {
        return startSizeArr;
    }

    public int[] getEndSizeArr() {
        return endSizeArr;
    }

    public void addBos() {
        ViterbiNode bosNode = new ViterbiNode(-1, BOS, 0, 0, 0, -1, ViterbiNode.Type.KNOWN);
        addNode(bosNode, 0, 1);
    }

    public void addEos() {
        ViterbiNode eosNode = new ViterbiNode(-1, EOS, 0, 0, 0, dimension - 1, ViterbiNode.Type.KNOWN);
        addNode(eosNode, dimension - 1, 0);
    }

    void addNode(ViterbiNode node, int start, int end) {
        addNodeToArray(node, start, getStartIndexArr(), getStartSizeArr());
        addNodeToArray(node, end, getEndIndexArr(), getEndSizeArr());
    }

    private void addNodeToArray(final ViterbiNode node, final int index, ViterbiNode[][] arr, int[] sizes) {
        int count = sizes[index];

        expandIfNeeded(index, arr, count);

        arr[index][count] = node;
        sizes[index] = count + 1;
    }

    private void expandIfNeeded(final int index, ViterbiNode[][] arr, final int count) {
        if (count == 0) {
            arr[index] = new ViterbiNode[10];
        }

        if (arr[index].length <= count) {
            arr[index] = extendArray(arr[index]);
        }
    }

    private ViterbiNode[] extendArray(ViterbiNode[] array) {
        ViterbiNode[] newArray = new ViterbiNode[array.length * 2];
        System.arraycopy(array, 0, newArray, 0, array.length);
        return newArray;
    }

    boolean tokenEndsWhereCurrentTokenStarts(int startIndex) {
        return getEndSizeArr()[startIndex + 1] != 0;
    }
}
