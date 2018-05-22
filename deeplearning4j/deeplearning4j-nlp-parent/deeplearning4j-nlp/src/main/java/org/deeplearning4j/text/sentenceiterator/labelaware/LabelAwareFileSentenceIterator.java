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

package org.deeplearning4j.text.sentenceiterator.labelaware;

import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 *
 * Label aware sentence iterator
 *
 * @author Adam Gibson
 */
public class LabelAwareFileSentenceIterator extends FileSentenceIterator implements LabelAwareSentenceIterator {
    /**
     * Takes a single file or directory
     *
     * @param preProcessor the sentence pre processor
     * @param file         the file or folder to iterate over
     */
    public LabelAwareFileSentenceIterator(SentencePreProcessor preProcessor, File file) {
        super(preProcessor, file);
    }

    public LabelAwareFileSentenceIterator(File dir) {
        super(dir);
    }

    @Override
    public String currentLabel() {
        return currentFile.getParentFile().getName();
    }

    @Override
    public List<String> currentLabels() {
        return Arrays.asList(currentFile.getParentFile().getName());
    }
}
