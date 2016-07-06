/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.canova.api.records.reader;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import org.canova.api.records.listener.RecordListener;

/**
 * Manages record listeners.
 *
 * @author saudet
 */
public abstract class BaseRecordReader implements RecordReader {

    protected Collection<RecordListener> listeners = new ArrayList<>();

    /** Invokes {@link RecordListener#recordRead(RecordReader, Object)} on all listeners. */
    protected void invokeListeners(Object record) {
        for(RecordListener listener : listeners) {
            listener.recordRead(this, record);
        }
    }

    @Override
    public Collection<RecordListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(Collection<RecordListener> listeners) {
        this.listeners = listeners;
    }

    @Override
    public void setListeners(RecordListener... listeners) {
        Collection<RecordListener> cListeners = new ArrayList<>();
        Collections.addAll(cListeners, listeners);
        setListeners(cListeners);
    }
}
