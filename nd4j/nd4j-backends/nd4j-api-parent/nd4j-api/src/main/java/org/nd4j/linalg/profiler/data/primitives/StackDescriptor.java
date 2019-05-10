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

package org.nd4j.linalg.profiler.data.primitives;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

/**
 * @author raver119
 */
@Slf4j
public class StackDescriptor {
    @Getter
    protected StackTraceElement stackTrace[];

    public StackDescriptor(@NonNull StackTraceElement stack[]) {
        // we cut off X first elements from stack, because they belong to profiler
        // basically, we just want to make sure, no profiler-related code is mentioned in stack trace
        int start = 0;
        for (; start < stack.length; start++) {
            if (stack[start].getClassName().contains("DefaultOpExecutioner"))
                break;
        }

        // in tests it's quite possible to have no DefaultOpExecutioner calls being used
        if (start == stack.length) {
            ;
            for (start = 0; start < stack.length; start++) {
                if (!stack[start + 1].getClassName().contains("OpProfiler")
                                && !stack[start + 1].getClassName().contains("StackAggregator"))
                    break;
            }
        } else {
            for (; start < stack.length; start++) {
                if (!stack[start].getClassName().contains("DefaultOpExecutioner"))
                    break;
            }
        }

        for (; start < stack.length; start++) {
            if (!stack[start].getClassName().contains("OpProfiler"))
                break;
        }

        this.stackTrace = Arrays.copyOfRange(stack, start, stack.length);
        ArrayUtils.reverse(this.stackTrace);
    }

    public String getEntryName() {
        return getElementName(0);
    }

    public String getElementName(int idx) {
        return stackTrace[idx].getClassName() + "." + stackTrace[idx].getMethodName() + ":"
                        + stackTrace[idx].getLineNumber();
    }

    public int size() {
        return stackTrace.length;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        StackDescriptor that = (StackDescriptor) o;

        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        return Arrays.equals(stackTrace, that.stackTrace);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(stackTrace);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Stack trace: \n");

        for (int i = 0; i < size(); i++) {
            builder.append("         ").append(i).append(": ").append(getElementName(i)).append("\n");
        }

        return builder.toString();
    }
}
