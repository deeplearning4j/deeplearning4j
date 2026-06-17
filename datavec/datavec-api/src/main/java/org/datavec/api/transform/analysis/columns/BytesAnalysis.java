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

package org.datavec.api.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

@AllArgsConstructor
@Builder
@Data
@NoArgsConstructor //For Jackson deserialization
public class BytesAnalysis implements ColumnAnalysis {

    private long countTotal;
    private long countNull;
    private long countZeroLength;
    private int minNumBytes;
    private int maxNumBytes;

    @Override
    public String toString() {
        return "BytesAnalysis()";
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Bytes;
    }

}
