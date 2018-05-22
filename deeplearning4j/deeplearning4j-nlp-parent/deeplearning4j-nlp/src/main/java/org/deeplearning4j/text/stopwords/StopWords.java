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

package org.deeplearning4j.text.stopwords;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.List;

/**
 * Loads stop words from the class path
 * @author Adam Gibson
 *
 */
public class StopWords {

    private static List<String> stopWords;

    private StopWords() {}

    @SuppressWarnings("unchecked")
    public static List<String> getStopWords() {

        try {
            if (stopWords == null)
                stopWords = IOUtils.readLines(new ClassPathResource("/stopwords.txt").getInputStream());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return stopWords;
    }

}
