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

package org.nd4j.linalg.profiler.data.primitives;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.common.com.scalified.tree.TraversalAction;
import org.nd4j.common.com.scalified.tree.TreeNode;
import org.nd4j.common.com.scalified.tree.multinode.LinkedMultiTreeNode;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class StackTree {

    private TreeNode<StackTraceElement> root;
    private TreeNode<StackTraceElement> lastNodeAdded;
    protected AtomicLong eventsCount = new AtomicLong(0);
    @Getter
    protected StackDescriptor lastDescriptor;

    public StackTree() {
        //

    }

    public String renderTree(boolean displayCounts) {
        StringBuilder builder = new StringBuilder();

        TraversalAction<TreeNode<StackTraceElement>> action = new TraversalAction<TreeNode<StackTraceElement>>() {
            @Override
            public void perform(TreeNode<StackTraceElement> node) {
                builder.append(StringUtils.repeat('\t',node.level()));
                builder.append(node.data().toString() + "\n");
            }

            @Override
            public boolean isCompleted() {
                return false; // return true in order to stop traversing
            }
        };


        root.traversePreOrder(action);
        return builder.toString();
    }


    public void consumeStackTrace(@NonNull StackDescriptor descriptor, long increment) {
        eventsCount.incrementAndGet();

        lastDescriptor = descriptor;

        if(root == null) {
            root = new LinkedMultiTreeNode<>(descriptor.getStackTrace()[0]);
            lastNodeAdded = root;
        }
        //traverse the stack trace looking for the node first
        //linking each element from the previous index
        //we can't just add it to the root
        //because we need to traverse the stack trace


        for(int i = 1; i < descriptor.getStackTrace().length; i++) {
            StackTraceElement element = descriptor.getStackTrace()[i];
            TreeNode<StackTraceElement> node = root.find(element);
            if(node == null) {
                node = new LinkedMultiTreeNode<>(element);
                lastNodeAdded.add(node);
                lastNodeAdded = node;
            }
        }
    }

    public long getTotalEventsNumber() {
        return eventsCount.get();
    }

    public int getUniqueBranchesNumber() {
        return 1;
    }

    public void reset() {
        root = null;
        eventsCount.set(0);
    }
}
