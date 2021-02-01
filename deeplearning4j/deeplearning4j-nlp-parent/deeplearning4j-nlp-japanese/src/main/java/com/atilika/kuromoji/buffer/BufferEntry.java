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
package com.atilika.kuromoji.buffer;

import java.util.ArrayList;
import java.util.List;

public class BufferEntry {

    public List<Short> tokenInfo = new ArrayList<>();
    public List<Integer> features = new ArrayList<>();
    public List<Byte> posInfo = new ArrayList<>();

    public short[] tokenInfos; // left id, right id, word cost values
    public int[] featureInfos; // references to string features
    public byte[] posInfos; // part-of-speech tag values

}
