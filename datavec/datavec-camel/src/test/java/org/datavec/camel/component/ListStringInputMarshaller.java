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

package org.datavec.camel.component;

import org.apache.camel.Exchange;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.ListStringSplit;

import java.util.List;

/**
 * Marshals List<List<String>>
 *
 *     @author Adam Gibson
 */
public class ListStringInputMarshaller implements DataVecMarshaller {
    /**
     * @param exchange
     * @return
     */
    @Override
    public InputSplit getSplit(Exchange exchange) {
        List<List<String>> data = (List<List<String>>) exchange.getIn().getBody();
        InputSplit listSplit = new ListStringSplit(data);
        return listSplit;
    }
}
