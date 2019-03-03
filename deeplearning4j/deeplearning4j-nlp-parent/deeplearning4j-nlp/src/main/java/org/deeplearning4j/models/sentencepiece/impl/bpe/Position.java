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

package org.deeplearning4j.models.sentencepiece.impl.bpe;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.val;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class Position {
    private long sid;
    private long left;
    private long right;


    public static Position decodePosition(long position) {
        val p = new Position();

        p.sid = position >> 32L;
        p.left = (position >> 16) & 0xffff;
        p.right = position & 0xffff;

        return p;
    }

    public static long encodePosition(Position p) {
        return encodePosition(p.sid, p.left, p.right);
    }

    public static long encodePosition(int sid, int left, int right) {
        long s = (long) sid;
        long l = (long) left;
        long r = (long) right;
        return encodePosition(s, l, r);
    }

    public static long encodePosition(long s, long l, long r) {
        val n = (s << 32 | (l << 16 | r));
        return n;
    }
}
