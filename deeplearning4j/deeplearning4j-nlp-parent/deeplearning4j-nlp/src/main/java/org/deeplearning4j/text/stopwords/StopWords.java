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

package org.deeplearning4j.text.stopwords;

import org.apache.commons.io.IOUtils;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;
import java.util.List;

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
