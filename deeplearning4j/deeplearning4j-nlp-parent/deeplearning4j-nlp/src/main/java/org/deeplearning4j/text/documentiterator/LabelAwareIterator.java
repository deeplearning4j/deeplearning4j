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

package org.deeplearning4j.text.documentiterator;

import java.util.Iterator;

/**
 * This simple iterator interface assumes, that all documents are packed into strings OR into references to VocabWords.
 * Basic idea is: for tasks like ParagraphVectors we need unified interface for reading Sentences (read: lines of text) or Documents (read: set of lines) with label support.
 *
 * There's 2 interoperbility implementations of this interfaces: SentenceIteratorConverter and DocumentIteratorConverter.
 * After conversion is done, they can be wrapped by BasicLabelAwareIterator, that accepts all 5 current interfaces (including this one) as source for labelled documents.
 * This way 100% backward compatibility is provided, as well as additional functionality is delivered via LabelsSource.
 *
 * @author raver119@gmail.com
 */
public interface LabelAwareIterator extends Iterator<LabelledDocument> {

    boolean hasNextDocument();

    LabelledDocument nextDocument();

    void reset();

    LabelsSource getLabelsSource();

    void shutdown();
}
