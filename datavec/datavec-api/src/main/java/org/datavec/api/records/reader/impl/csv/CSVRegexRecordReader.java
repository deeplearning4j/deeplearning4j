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

package org.datavec.api.records.reader.impl.csv;

import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A CSVRecordReader that can split
 * each column into additional columns using regexs.
 *
 * @author saudet
 */
public class CSVRegexRecordReader extends CSVRecordReader {

    protected String[] regexs = null;
    protected Pattern[] patterns = null;
    protected String delimiter;
    protected String quote;

    /**
     * Skip lines, use delimiter, strip quotes, and parse each column with a regex
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     * @param quote the quote to strip
     * @param regexs the regexs to parse columns with
     */
    public CSVRegexRecordReader(int skipNumLines, String delimiter, String quote, String[] regexs) {
        super(skipNumLines);
        this.delimiter = delimiter;
        this.quote = quote;
        this.regexs = regexs;
        if (regexs != null) {
            patterns = new Pattern[regexs.length];
            for (int i = 0; i < regexs.length; i++) {
                if (regexs[i] != null) {
                    patterns[i] = Pattern.compile(regexs[i]);
                }
            }
        }
    }

    protected List<Writable> parseLine(String line) {
        String[] split = line.split(delimiter, -1);
        List<Writable> ret = new ArrayList<>();
        for (int i = 0; i < split.length; i++) {
            String s = split[i];
            if (quote != null && s.startsWith(quote) && s.endsWith(quote)) {
                int n = quote.length();
                s = s.substring(n, s.length() - n).replace(quote + quote, quote);
            }
            if (regexs != null && regexs[i] != null) {
                Matcher m = patterns[i].matcher(s);
                if (m.matches()) {
                    for (int j = 1; j <= m.groupCount(); j++) { //Note: Matcher.group(0) is the entire sequence; we only care about groups 1 onward
                        ret.add(new Text(m.group(j)));
                    }
                } else {
                    throw new IllegalStateException("Invalid line: value does not match regex (regex=\"" + regexs[i]
                                    + "\"; value=\"" + s + "\"");
                }
            } else {
                ret.add(new Text(s));
            }
        }
        return ret;
    }

}
