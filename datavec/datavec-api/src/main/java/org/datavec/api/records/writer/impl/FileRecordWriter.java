/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.writer.impl;


import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.Partitioner;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

/**
 * Write to files.
 * <p>
 * To set the path and configuration via configuration:
 * writeTo: org.datavec.api.records.writer.path
 * <p>
 * This is the path used to write to
 *
 * @author Adam Gibson
 */
public abstract class FileRecordWriter implements RecordWriter {

    public static final Charset DEFAULT_CHARSET = StandardCharsets.UTF_8;

    protected DataOutputStream out;
    public final static String NEW_LINE = "\n";

    protected Charset encoding = DEFAULT_CHARSET;

    protected Partitioner partitioner;

    protected Configuration conf;

    public FileRecordWriter() {
    }

    @Override
    public void initialize(InputSplit inputSplit, Partitioner partitioner) throws Exception {
        partitioner.init(inputSplit);
        out = new DataOutputStream(partitioner.currentOutputStream());
        this.partitioner = partitioner;

    }

    @Override
    public void initialize(Configuration configuration, InputSplit split, Partitioner partitioner) throws Exception {
        setConf(configuration);
        partitioner.init(configuration, split);
        initialize(split, partitioner);
    }

    @Override
    public void close() {
        if (out != null) {
            try {
                out.flush();
                out.close();
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }

        }
    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }
}
