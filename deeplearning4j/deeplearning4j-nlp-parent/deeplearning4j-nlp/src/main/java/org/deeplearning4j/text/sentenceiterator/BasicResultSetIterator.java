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

import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * Primitive iterator over a SQL ResultSet
 *
 * Please note: for reset functionality, the underlying JDBC ResultSet must not be of TYPE_FORWARD_ONLY
 * To achieve this using postgres you can make your query using: connection.prepareStatement(sql,ResultSet.TYPE_SCROLL_INSENSITIVE,ResultSet.CONCUR_READ_ONLY);
 *
 * This class is designed in a similar fashion to org.deeplearning4j.text.sentenceiterator.BasicLineIterator
 *
 * @author Brad Heap nzv8fan@gmail.com
 */
public class BasicResultSetIterator implements SentenceIterator {

    private ResultSet rs;
    private String columnName;

    private SentencePreProcessor preProcessor;

    private boolean nextCalled; // we use this to ensure that next is only called once by hasNext() to ensure we don't skip over data
    private boolean resultOfNext;

    public BasicResultSetIterator(ResultSet rs, String columnName) {
        this.rs = rs;
        this.columnName = columnName;

        this.nextCalled = false;
        this.resultOfNext = false;
    }

    public synchronized String nextSentence() {
        try {
            if (!nextCalled) { // move onto the next row if we haven't yet
                rs.next();
            } else {
                nextCalled = false; // reset that next has been called for next time we call nextSentence() or hasNext()
            }
            return (preProcessor != null) ? this.preProcessor.preProcess(rs.getString(columnName))
                            : rs.getString(columnName);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public synchronized boolean hasNext() {
        try {
            if (!nextCalled) {
                resultOfNext = rs.next();
                nextCalled = true;
            }
            return resultOfNext;
        } catch (SQLException e) {
            return false;
        }
    }

    public synchronized void reset() {
        try {
            rs.beforeFirst();
            nextCalled = false;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public void finish() {
        try {
            rs.close();
        } catch (SQLException e) {
            // do nothing here
        }
    }

    public SentencePreProcessor getPreProcessor() {
        return preProcessor;
    }

    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }
}
