/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.records.reader.impl.jackson;

import java.util.List;

import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.databind.ObjectMapper;

public class JacksonLineRecordReader extends LineRecordReader {

    private FieldSelection selection;
    private ObjectMapper mapper;

	public JacksonLineRecordReader(FieldSelection selection, ObjectMapper mapper) {
		this.selection = selection;
		this.mapper = mapper;
	}

    @Override
    public List<Writable> next() {
        Text t = (Text) super.next().iterator().next();
        String val = t.toString();
        return parseLine(val);
    }
    
    protected List<Writable> parseLine(String line) {
	    return JacksonReaderUtils.parseRecord(line, selection, mapper);
    }
    
    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException();
    }
}
