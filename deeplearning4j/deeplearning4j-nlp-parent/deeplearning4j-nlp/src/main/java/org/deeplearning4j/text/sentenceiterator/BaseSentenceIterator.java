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

package org.deeplearning4j.text.sentenceiterator;

public abstract class BaseSentenceIterator implements SentenceIterator {

    protected SentencePreProcessor preProcessor;



    public BaseSentenceIterator(SentencePreProcessor preProcessor) {
        super();
        this.preProcessor = preProcessor;
    }

    public BaseSentenceIterator() {
        super();
    }

    public SentencePreProcessor getPreProcessor() {
        return preProcessor;
    }

    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public void finish() {
        //No-op
    }



}
