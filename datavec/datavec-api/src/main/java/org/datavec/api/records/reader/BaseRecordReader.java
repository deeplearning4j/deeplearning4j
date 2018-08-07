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

package org.datavec.api.records.reader;

import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Manages record listeners.
 *
 * @author saudet
 */
public abstract class BaseRecordReader implements RecordReader {

    protected List<RecordListener> listeners = new ArrayList<>();

    /** Invokes {@link RecordListener#recordRead(RecordReader, Object)} on all listeners. */
    protected void invokeListeners(Object record) {
        for (RecordListener listener : listeners) {
            listener.recordRead(this, record);
        }
    }

    @Override
    public List<RecordListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        this.listeners = (listeners instanceof List ? (List<RecordListener>) listeners : new ArrayList<>(listeners));
    }

    @Override
    public void setListeners(RecordListener... listeners) {
        setListeners(Arrays.asList(listeners));
    }


    @Override
    public boolean batchesSupported() {
        return false;
    }

    @Override
    public List<List<Writable>> next(int num) {
        throw new UnsupportedOperationException();
    }
}
