/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */
package org.datavec.api.records.listener.impl;

import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A record listener that logs every record to be read or written.
 *
 * @author saudet
 */
public class LogRecordListener implements RecordListener {
    private static final Logger log = LoggerFactory.getLogger(LogRecordListener.class);
    private boolean invoked = false;

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        this.invoked = true;
    }

    @Override
    public void recordRead(RecordReader reader, Object record) {
        invoke();
        log.info("Reading " + record);
    }

    @Override
    public void recordWrite(RecordWriter writer, Object record) {
        invoke();
        log.info("Writing " + record);
    }
}
