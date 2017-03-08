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

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.List;


public class WordConverter {

    private List<String> sentences = new ArrayList<>();
    private Word2Vec vec;
    private List<Window> windows;

    public WordConverter(List<String> sentences, Word2Vec vec) {
        this.sentences = sentences;
        this.vec = vec;
    }

    public static INDArray toInputMatrix(List<Window> windows, Word2Vec vec) {
        int columns = vec.lookupTable().layerSize() * vec.getWindow();
        int rows = windows.size();
        INDArray ret = Nd4j.create(rows, columns);
        for (int i = 0; i < rows; i++) {
            ret.putRow(i, WindowConverter.asExampleMatrix(windows.get(i), vec));
        }
        return ret;
    }


    public INDArray toInputMatrix() {
        List<Window> windows = allWindowsForAllSentences();
        return toInputMatrix(windows, vec);
    }



    public static INDArray toLabelMatrix(List<String> labels, List<Window> windows) {
        int columns = labels.size();
        INDArray ret = Nd4j.create(windows.size(), columns);
        for (int i = 0; i < ret.rows(); i++) {
            ret.putRow(i, FeatureUtil.toOutcomeVector(labels.indexOf(windows.get(i).getLabel()), labels.size()));
        }
        return ret;
    }

    public INDArray toLabelMatrix(List<String> labels) {
        List<Window> windows = allWindowsForAllSentences();
        return toLabelMatrix(labels, windows);
    }

    private List<Window> allWindowsForAllSentences() {
        if (windows != null)
            return windows;
        windows = new ArrayList<>();
        for (String s : sentences)
            if (!s.isEmpty())
                windows.addAll(Windows.windows(s));
        return windows;
    }



}
