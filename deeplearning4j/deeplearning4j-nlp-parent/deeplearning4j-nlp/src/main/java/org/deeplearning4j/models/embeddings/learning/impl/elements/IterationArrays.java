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
package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.Data;

@Data
public class IterationArrays {
    private int itemSize;
    private int maxCols;
    private int maxWinWordsCols;
    int[][] indicesArr;
    int[][] codesArr;
    long[] randomValues;
    int[] ngStarters;
    double[] alphas;
    int[] targets;
    int[][] inputWindowWordsArr;
    int[][] inputWindowWordStatuses;
    int[] currentWindowIndexes;
    int[] numLabels;

    public IterationArrays(int itemSize, int maxCols,int maxWinWordsCols) {
        this.maxWinWordsCols = maxWinWordsCols;
        this.itemSize = itemSize;
        this.maxCols = maxCols;
        indicesArr = new int[itemSize][maxCols];
        codesArr = new int[itemSize][maxCols];
        currentWindowIndexes = new int[itemSize];
        inputWindowWordsArr = new int[itemSize][maxWinWordsCols];
        inputWindowWordStatuses = new int[itemSize][maxWinWordsCols];
        randomValues = new long[itemSize];
        ngStarters = new int[itemSize];
        alphas = new double[itemSize];
        targets = new int[itemSize];
        numLabels = new int[itemSize];
        initCodes();
    }

    public IterationArrays(int itemSize, int maxCols) {
        this(itemSize,maxCols,0);
    }

    /**
     * USed to initialize codes to the -1 default value.
     * This method should be called when reusing an iteration arrays object.
     */
    public void initCodes() {
        for (int i = 0; i < codesArr.length; i++) {
            for (int j = 0; j < codesArr[0].length; j++) {
                codesArr[i][j] = -1;
            }
        }
    }

}
