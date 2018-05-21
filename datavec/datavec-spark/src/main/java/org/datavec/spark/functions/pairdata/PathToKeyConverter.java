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

package org.datavec.spark.functions.pairdata;

import java.io.Serializable;

/** PathToKeyConverter: Used to match up files based on their file names, for PairSequenceRecordReaderBytesFunction
 * For example, suppose we have files "/features_0.csv" and "/labels_0.csv", map both to same key: "0"
 */
public interface PathToKeyConverter extends Serializable {

    /**Determine the key from the file path
     * @param path Input path
     * @return Key for the file
     */
    String getKey(String path);

}
