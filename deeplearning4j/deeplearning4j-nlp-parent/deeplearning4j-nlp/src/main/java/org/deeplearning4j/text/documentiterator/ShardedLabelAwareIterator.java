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

package org.deeplearning4j.text.documentiterator;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.shade.guava.collect.Lists;

import java.util.List;

/**
 * The ShardedLabelAwareIterator class is an implementation of the LabelAwareIterator interface.
 * It wraps a given LabelAwareIterator (subIterator) and splits its documents into smaller documents (shards)
 * to help manage memory usage. This is particularly useful when dealing with large text documents.
 * The documentSizeLimit field determines the maximum number of tokens per shard.
 * The tokenizerFactory field is used to tokenize the documents before sharding them.
 * The docBatches field stores the shards, and the currentBatch field keeps track of the current shard index.
 *
 * @author Adam Gibson
 */
public class ShardedLabelAwareIterator implements LabelAwareIterator {
    // Instance variables
    private LabelAwareIterator subIterator;
    private int documentSizeLimit;
    private TokenizerFactory tokenizerFactory;
    private List<List<String>> docBatches;
    private int currentBatch = 0;

    // Constructor
    public ShardedLabelAwareIterator(LabelAwareIterator subIterator, TokenizerFactory tokenizerFactory, int documentSizeLimit) {
        this.subIterator = subIterator;
        this.documentSizeLimit = documentSizeLimit;
        this.tokenizerFactory = tokenizerFactory;
    }

    // Splits a document into smaller documents (shards) based on the documentSizeLimit
    private void shardDocument(LabelledDocument document) {
        Tokenizer tokenizer = tokenizerFactory.create(document.getContent());
        this.docBatches = Lists.partition(tokenizer.getTokens(), documentSizeLimit);
        currentBatch = 0;
    }

    // Checks if there are more documents available
    @Override
    public boolean hasNextDocument() {
        LabelledDocument nextDoc = nextDocument();
        if (nextDoc != null) {
            currentBatch--; // Revert the currentBatch increment in the nextDocument() method
        }
        return nextDoc != null;
    }

    // Retrieves the next document from the iterator
    @Override
    public LabelledDocument nextDocument() {
        while (docBatches == null || currentBatch >= docBatches.size() || (docBatches != null && docBatches.isEmpty())) {
            if (!subIterator.hasNextDocument()) {
                return null; // Return null if no more documents are available
            }

            LabelledDocument document = subIterator.nextDocument();
            shardDocument(document);
        }


        if (currentBatch < docBatches.size()) {
            LabelledDocument document = new LabelledDocument();
            document.setLabels(subIterator.getLabelsSource().getLabels());
            document.setContent(String.join(" ", docBatches.get(currentBatch)));
            currentBatch++;
            return document;
        } else {
            throw new IllegalStateException("No more documents");
        }
    }

    // Resets the iterator
    @Override
    public void reset() {
        subIterator.reset();
        this.docBatches = null;
        currentBatch = 0;
    }

    // Returns the LabelsSource from the wrapped iterator
    @Override
    public LabelsSource getLabelsSource() {
        return subIterator.getLabelsSource();
    }

    // Empty method for shutting down the iterator (not needed in this case)
    @Override
    public void shutdown() {
    }

    // Alias for hasNextDocument(), checks if there are more documents available
    @Override
    public boolean hasNext() {
        return hasNextDocument();
    }

    // Alias for nextDocument(), retrieves the next document from the iterator
    @Override
    public LabelledDocument next() {
        return nextDocument();
    }

    public LabelAwareIterator getSubIterator() {
        return subIterator;
    }

    public void setSubIterator(LabelAwareIterator subIterator) {
        this.subIterator = subIterator;
    }

    public int getDocumentSizeLimit() {
        return documentSizeLimit;
    }

    public void setDocumentSizeLimit(int documentSizeLimit) {
        this.documentSizeLimit = documentSizeLimit;
    }

    public TokenizerFactory getTokenizerFactory() {
        return tokenizerFactory;
    }

    public void setTokenizerFactory(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
    }

    public List<List<String>> getDocBatches() {
        return docBatches;
    }

    public void setDocBatches(List<List<String>> docBatches) {
        this.docBatches = docBatches;
    }

    public int getCurrentBatch() {
        return currentBatch;
    }

    public void setCurrentBatch(int currentBatch) {
        this.currentBatch = currentBatch;
    }
}