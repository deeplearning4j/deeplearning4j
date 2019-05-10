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

package org.deeplearning4j.iterator.provider;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.collection.CompactHeapStringList;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Iterate over a set of sentences/documents, where the sentences are to be loaded (as required) from the provided files.
 *
 * @author Alex Black
 */
public class FileLabeledSentenceProvider implements LabeledSentenceProvider {

    private final int totalCount;
    private final List<String> filePaths;
    private final int[] fileLabelIndexes;
    private final Random rng;
    private final int[] order;
    private final List<String> allLabels;

    private int cursor = 0;

    /**
     * @param filesByLabel Key: label. Value: list of files for that label
     */
    public FileLabeledSentenceProvider(Map<String, List<File>> filesByLabel) {
        this(filesByLabel, new Random());
    }

    /**
     *
     * @param filesByLabel Key: label. Value: list of files for that label
     * @param rng          Random number generator. May be null.
     */
    public FileLabeledSentenceProvider(@NonNull Map<String, List<File>> filesByLabel, Random rng) {
        int totalCount = 0;
        for (List<File> l : filesByLabel.values()) {
            totalCount += l.size();
        }
        this.totalCount = totalCount;

        this.rng = rng;
        if (rng == null) {
            order = null;
        } else {
            order = new int[totalCount];
            for (int i = 0; i < totalCount; i++) {
                order[i] = i;
            }

            MathUtils.shuffleArray(order, rng);
        }

        allLabels = new ArrayList<>(filesByLabel.keySet());
        Collections.sort(allLabels);

        Map<String, Integer> labelsToIdx = new HashMap<>();
        for (int i = 0; i < allLabels.size(); i++) {
            labelsToIdx.put(allLabels.get(i), i);
        }

        filePaths = new CompactHeapStringList();
        fileLabelIndexes = new int[totalCount];
        int position = 0;
        for (Map.Entry<String, List<File>> entry : filesByLabel.entrySet()) {
            int labelIdx = labelsToIdx.get(entry.getKey());
            for (File f : entry.getValue()) {
                filePaths.add(f.getPath());
                fileLabelIndexes[position] = labelIdx;
                position++;
            }
        }
    }

    @Override
    public boolean hasNext() {
        return cursor < totalCount;
    }

    @Override
    public Pair<String, String> nextSentence() {
        int idx;
        if (rng == null) {
            idx = cursor++;
        } else {
            idx = order[cursor++];
        }
        File f = new File(filePaths.get(idx));
        String label = allLabels.get(fileLabelIndexes[idx]);

        String sentence;
        try {
            sentence = FileUtils.readFileToString(f);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return new Pair<>(sentence, label);
    }

    @Override
    public void reset() {
        cursor = 0;
        if (rng != null) {
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public int totalNumSentences() {
        return totalCount;
    }

    @Override
    public List<String> allLabels() {
        return allLabels;
    }

    @Override
    public int numLabelClasses() {
        return allLabels.size();
    }
}
