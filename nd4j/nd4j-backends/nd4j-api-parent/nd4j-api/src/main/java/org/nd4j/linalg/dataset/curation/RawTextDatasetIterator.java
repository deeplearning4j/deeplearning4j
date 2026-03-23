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

package org.nd4j.linalg.dataset.curation;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Dataset iterator that reads raw text files, chunks with overlap,
 * and produces MultiDataSets with input_ids and labels for causal LM training.
 *
 * All tokens are trained on (no masking), making this suitable for
 * continued pretraining / domain adaptation rather than instruction tuning.
 *
 * Requires a tokenizer function to convert text to token IDs.
 *
 * Adam Gibson
 */
@Slf4j
public class RawTextDatasetIterator implements MultiDataSetIterator {

    @Getter
    private final int chunkSize;

    @Getter
    private final int chunkOverlap;

    @Getter
    private final int batchSize;

    private final Function<String, long[]> tokenizer;
    private final List<File> textFiles;

    private MultiDataSetPreProcessor preProcessor;

    // Internal state
    private int currentFileIdx = 0;
    private long[] currentTokens;
    private int currentTokenPos = 0;
    private boolean exhausted = false;

    /**
     * @param textFiles    List of text files to read
     * @param tokenizer    Function that converts text to token IDs
     * @param chunkSize    Sequence length for each chunk
     * @param chunkOverlap Number of overlapping tokens between chunks
     * @param batchSize    Number of chunks per batch
     */
    public RawTextDatasetIterator(List<File> textFiles, Function<String, long[]> tokenizer,
                                   int chunkSize, int chunkOverlap, int batchSize) {
        this.textFiles = textFiles;
        this.tokenizer = tokenizer;
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.batchSize = batchSize;

        loadNextFile();
    }

    @Override
    public MultiDataSet next(int num) {
        List<long[]> chunks = new ArrayList<>();

        while (chunks.size() < num && !exhausted) {
            long[] chunk = nextChunk();
            if (chunk != null) {
                chunks.add(chunk);
            }
        }

        if (chunks.isEmpty()) {
            return null;
        }

        // Build input_ids and labels
        // For causal LM: labels = input_ids shifted left by 1
        int actualBatch = chunks.size();
        INDArray inputIds = Nd4j.create(DataType.INT64, actualBatch, chunkSize);
        INDArray labels = Nd4j.create(DataType.INT64, actualBatch, chunkSize);

        for (int b = 0; b < actualBatch; b++) {
            long[] chunk = chunks.get(b);
            for (int t = 0; t < chunkSize; t++) {
                inputIds.putScalar(b, t, chunk[t]);
                // Labels: shifted by 1 (predict next token)
                if (t < chunkSize - 1) {
                    labels.putScalar(b, t, chunk[t + 1]);
                } else {
                    labels.putScalar(b, t, -100); // ignore index for last position
                }
            }
        }

        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{inputIds},
                new INDArray[]{labels}
        );

        if (preProcessor != null) {
            preProcessor.preProcess(mds);
        }

        return mds;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public boolean hasNext() {
        return !exhausted;
    }

    @Override
    public void reset() {
        currentFileIdx = 0;
        currentTokens = null;
        currentTokenPos = 0;
        exhausted = false;
        loadNextFile();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    private long[] nextChunk() {
        while (!exhausted) {
            if (currentTokens != null && currentTokenPos + chunkSize <= currentTokens.length) {
                long[] chunk = new long[chunkSize];
                System.arraycopy(currentTokens, currentTokenPos, chunk, 0, chunkSize);
                currentTokenPos += (chunkSize - chunkOverlap);
                return chunk;
            }

            // Need more tokens — load next file
            currentFileIdx++;
            if (!loadNextFile()) {
                exhausted = true;
                return null;
            }
        }
        return null;
    }

    private boolean loadNextFile() {
        while (currentFileIdx < textFiles.size()) {
            try {
                String text = new String(Files.readAllBytes(textFiles.get(currentFileIdx).toPath()),
                        StandardCharsets.UTF_8);
                currentTokens = tokenizer.apply(text);
                currentTokenPos = 0;

                if (currentTokens.length >= chunkSize) {
                    return true;
                }

                // File too short, skip
                log.debug("Skipping file {} (only {} tokens, need {})",
                        textFiles.get(currentFileIdx), currentTokens.length, chunkSize);
                currentFileIdx++;
            } catch (IOException e) {
                log.warn("Error reading file: {}", textFiles.get(currentFileIdx), e);
                currentFileIdx++;
            }
        }
        return false;
    }
}
