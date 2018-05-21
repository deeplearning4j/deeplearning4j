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

package org.datavec.api.records.reader.impl.misc;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.conf.Configuration;

/**
 * Record reader for libsvm format, which is closely
 * related to SVMLight format. Similar to scikit-learn
 * we use a single reader for both formats, so this class
 * is a subclass of SVMLightRecordReader.
 *
 * @see SVMLightRecordReader
 *
 * Further details on the format can be found at
 * - http://svmlight.joachims.org/
 * - http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html
 * - http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
 *
 * @author Adam Gibson     (original)
 * @author dave@skymind.io
 */
@Slf4j
public class LibSvmRecordReader extends SVMLightRecordReader {
    public LibSvmRecordReader() {
        super();
    }

    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
    }
}
