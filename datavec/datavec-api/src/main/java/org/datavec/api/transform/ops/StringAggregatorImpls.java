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

package org.datavec.api.transform.ops;

import lombok.Getter;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Groups useful {@link IAggregableReduceOp} utilities on Strings
 *
 * Created by huitseeker on 5/18/17.
 */
public class StringAggregatorImpls {

    private static abstract class AggregableStringReduce implements IAggregableReduceOp<String, Writable> {
        @Getter
        protected StringBuilder sb = new StringBuilder();
    }

    public static class AggregableStringAppend extends AggregableStringReduce {

        @Override
        public <W extends IAggregableReduceOp<String, Writable>> void combine(W accu) {
            if (accu instanceof AggregableStringAppend)
                sb.append(((AggregableStringAppend) accu).getSb());
            else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where" + this.getClass().getName() + " expected");
        }

        @Override
        public void accept(String s) {
            sb.append(s);
        }

        @Override
        public Writable get() {
            return new Text(sb.toString());
        }
    }

    public static class AggregableStringPrepend extends AggregableStringReduce {

        @Override
        public <W extends IAggregableReduceOp<String, Writable>> void combine(W accu) {
            if (accu instanceof AggregableStringPrepend)
                sb.append(((AggregableStringPrepend) accu).getSb());
            else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where" + this.getClass().getName() + " expected");
        }

        @Override
        public void accept(String s) {
            String rev = new StringBuilder(s).reverse().toString();
            sb.append(rev);
        }

        @Override
        public Writable get() {
            return new Text(sb.reverse().toString());
        }
    }

}
