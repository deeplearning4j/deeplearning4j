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

package org.deeplearning4j.text.sentenceiterator;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;

import java.io.*;

/**
 * Each line is a sentence
 *
 * @author Adam Gibson
 */
public class LineSentenceIterator extends BaseSentenceIterator {

    private InputStream file;
    private LineIterator iter;
    private File f;



    public LineSentenceIterator(File f) {
        if (!f.exists() || !f.isFile())
            throw new IllegalArgumentException("Please specify an existing file");
        try {
            this.f = f;
            this.file = new BufferedInputStream(new FileInputStream(f));
            iter = IOUtils.lineIterator(this.file, "UTF-8");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String nextSentence() {
        String line = iter.nextLine();
        if (preProcessor != null) {
            line = preProcessor.preProcess(line);
        }
        return line;
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public void reset() {
        try {
            if (file != null)
                file.close();
            if (iter != null)
                iter.close();
            this.file = new BufferedInputStream(new FileInputStream(f));
            iter = IOUtils.lineIterator(this.file, "UTF-8");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }


}
