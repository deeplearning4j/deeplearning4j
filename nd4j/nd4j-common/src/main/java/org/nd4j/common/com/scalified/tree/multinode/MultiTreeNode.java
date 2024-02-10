/*
 * Copyright 2016 Scalified <http://www.scalified.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.nd4j.common.com.scalified.tree.multinode;

import org.nd4j.common.com.scalified.tree.TreeNode;
import org.nd4j.common.com.scalified.tree.TreeNodeException;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;

/**
 * This class represents the K-ary (multiple node) tree data
 * structure
 * <h1>Definition</h1>
 * <p>
 * <b>K-ary tree</b> - tree, in which each node has no more than k subtrees
 *
 * @author shell
 * @version 1.0.0
 * @since 1.0.0
 */
public abstract class MultiTreeNode<T> extends TreeNode<T> {

	/**
	 * Creates an instance of this class
	 *
	 * @param data data to store in the current tree node
	 */
	public MultiTreeNode(T data) {
		super(data);
	}

	/**
	 * Adds the collection of the subtrees with all of theirs descendants
	 * to the current tree node
	 * <p>
	 * Checks whether this tree node was changed as a result of the call
	 *
	 * @param subtrees collection of the subtrees with all of their
	 *                 descendants
	 * @return {@code true} if this tree node was changed as a
	 *         result of the call; {@code false} otherwise
	 */
	public boolean addSubtrees(Collection<? extends MultiTreeNode<T>> subtrees) {
		if (areAllNulls(subtrees)) {
			return false;
		}
		for (MultiTreeNode<T> subtree : subtrees) {
			linkParent(subtree, this);
			if (!add(subtree)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Returns the collection of nodes, which have the same parent
	 * as the current node; {@link Collections#emptyList()} if the current
	 * tree node is root or if the current tree node has no subtrees
	 *
	 * @return collection of nodes, which have the same parent as
	 *         the current node; {@link Collections#emptyList()} if the
	 *         current tree node is root or if the current tree node has
	 *         no subtrees
	 */
	public Collection<? extends MultiTreeNode<T>> siblings() {
		if (isRoot()) {
			String message = String.format("Unable to find the siblings. The tree node %1$s is root", root());
			throw new TreeNodeException(message);
		}
		Collection<? extends TreeNode<T>> parentSubtrees = parent.subtrees();
		int parentSubtreesSize = parentSubtrees.size();
		if (parentSubtreesSize == 1) {
			return Collections.emptySet();
		}
		Collection<MultiTreeNode<T>> siblings = new HashSet<>(parentSubtreesSize - 1);
		for (TreeNode<T> parentSubtree : parentSubtrees) {
			if (!parentSubtree.equals(this)) {
				siblings.add((MultiTreeNode<T>) parentSubtree);
			}
		}
		return siblings;
	}

	/**
	 * Checks whether among the current tree node subtrees there are
	 * all of the subtrees from the specified collection
	 *
	 * @param subtrees collection of subtrees to be checked for containment
	 *                 within the current tree node subtrees
	 * @return {@code true} if among the current tree node subtrees
	 *         there are all of the subtrees from the specified collection;
	 *         {@code false} otherwise
	 */
	public boolean hasSubtrees(Collection<? extends MultiTreeNode<T>> subtrees) {
		if (isLeaf()
				|| areAllNulls(subtrees)) {
			return false;
		}
		for (MultiTreeNode<T> subtree : subtrees) {
			if (!this.hasSubtree(subtree)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Removes all of the collection's subtrees from the current tree node
	 * <p>
	 * Checks whether the current tree node was changed as a result of
	 * the call
	 *
	 * @param subtrees collection containing subtrees to be removed from the
	 *                 current tree node
	 * @return {@code true} if the current tree node was changed as a result
	 *         of the call; {@code false} otherwise
	 */
	public boolean dropSubtrees(Collection<? extends MultiTreeNode<T>> subtrees) {
		if (isLeaf()
				|| areAllNulls(subtrees)) {
			return false;
		}
		boolean result = false;
		for (MultiTreeNode<T> subtree : subtrees) {
			boolean currentResult = dropSubtree(subtree);
			if (!result && currentResult) {
				result = true;
			}
		}
		return result;
	}

}
