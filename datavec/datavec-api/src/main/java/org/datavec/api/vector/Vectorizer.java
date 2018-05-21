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

package org.datavec.api.vector;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.reader.RecordReader;

/**
 * Vectorizer of a particular type.
 * Meant for converting individual records to vectors
 *
 * @author Adam Gibson
 */
public interface Vectorizer<VECTOR_TYPE> {


    /**
     * Create a vector based on the given arguments
     * @param args the arguments to create a vector with
     * @return the created vector
     *
     */
    VECTOR_TYPE createVector(Object[] args);

    /**
     * Initialize based on a configuration
     * @param conf the configuration to use
     */
    void initialize(Configuration conf);

    /**
     * Fit based on a record reader
     * @param reader
     */
    void fit(RecordReader reader);

    /**
     * Fit based on a record reader
     * @param reader
     */
    VECTOR_TYPE fitTransform(RecordReader reader);


    /**
     * Fit based on a record reader
     * @param reader
     * @param callBack
     */
    void fit(RecordReader reader, RecordCallBack callBack);

    /**
     * Fit based on a record reader
     * @param reader
     * @param callBack
     */
    VECTOR_TYPE fitTransform(RecordReader reader, RecordCallBack callBack);

    /**
     * Transform a record in to a vector
     * @param record the record to write
     * @return
     */
    VECTOR_TYPE transform(Record record);


    /**
     * On record call back.
     * This allows for neat inheritance and polymorphism
     * for fit and fit/transform among other things
     */
    public static interface RecordCallBack {
        /**
         * The record callback
         * @param record
         */
        void onRecord(Record record);
    }


}
