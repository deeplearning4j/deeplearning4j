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

package org.datavec.api.records.reader.impl.misc;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.conf.Configuration;

/**
 * Record reader for libsvm format, which is closely
 * related to SVMLight format. Similar to scikit-learn
 * we use a single reader for both formats, so this class
 * is a subclass of SVMLightRecordReader.
 *
 * Further details on the format can be found at<br>
 * - <a href="http://svmlight.joachims.org/">http://svmlight.joachims.org/</a><br>
 * - <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html">http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html</a><br>
 * - <a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html">http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html</a>
 *
 * @see SVMLightRecordReader
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
