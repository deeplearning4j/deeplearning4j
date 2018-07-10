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

package org.ansj.app.summary;

import org.ansj.app.keyword.Keyword;
import org.ansj.app.summary.pojo.Summary;
import org.nlpcn.commons.lang.tire.SmartGetWord;
import org.nlpcn.commons.lang.tire.domain.SmartForest;

import java.util.List;

/**
 * 关键字标红，
 * 
 * @author ansj
 * 
 */
public class TagContent {

    private String beginTag, endTag;

    public TagContent(String beginTag, String endTag) {
        this.beginTag = beginTag;
        this.endTag = endTag;
    }

    public String tagContent(Summary summary) {
        return tagContent(summary.getKeyWords(), summary.getSummary());
    }

    public String tagContent(List<Keyword> keyWords, String content) {
        SmartForest<Double> sf = new SmartForest<>();
        for (Keyword keyWord : keyWords) {
            sf.add(keyWord.getName().toLowerCase(), keyWord.getScore());
        }

        SmartGetWord<Double> sgw = new SmartGetWord<>(sf, content.toLowerCase());

        int beginOffe = 0;
        String temp = null;
        StringBuilder sb = new StringBuilder();
        while ((temp = sgw.getFrontWords()) != null) {
            sb.append(content.substring(beginOffe, sgw.offe));
            sb.append(beginTag);
            sb.append(content.substring(sgw.offe, sgw.offe + temp.length()));
            sb.append(endTag);
            beginOffe = sgw.offe + temp.length();
        }

        if (beginOffe <= content.length() - 1) {
            sb.append(content.substring(beginOffe, content.length()));
        }

        return sb.toString();
    }

}
