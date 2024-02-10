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

import org.nd4j.common.com.scalified.tree.TraversalAction;
import org.nd4j.common.com.scalified.tree.TreeNode;
import org.nd4j.common.com.scalified.tree.TreeNodeException;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;

/**
 * Implementation of the K-ary (multi node) tree data structure,
 * based on the leftmost-child-right-sibling representation
 *
 * @author shell
 * @version 1.0.0
 * @since 1.0.0
 */
public class LinkedMultiTreeNode<T> extends MultiTreeNode<T> {

	/**
	 * Current UID of this object used for serialization
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * A reference to the first subtree tree node of the current tree node
	 */
	private LinkedMultiTreeNode<T> leftMostNode;

	/**
	 * A reference to the right sibling tree node of the current tree node
	 */
	private LinkedMultiTreeNode<T> rightSiblingNode;

	/**
	 * A reference to the last subtree node of the current tree node
	 * <p>
	 * Used to avoid the discovery of the last subtree node. As a result
	 * significantly optimized such operations like addition etc.
	 */
	private LinkedMultiTreeNode<T> lastSubtreeNode;

	/**
	 * Creates an instance of this class
	 *
	 * @param data data to store in the current tree node
	 */
	public LinkedMultiTreeNode(T data) {
		super(data);
	}

	/**
	 * Returns the collection of the child nodes of the current node
	 * with all of its proper descendants, if any
	 * <p>
	 * Returns {@link Collections#emptySet()} if the current node is leaf
	 *
	 * @return collection of the child nodes of the current node with
	 *         all of its proper descendants, if any;
	 *         {@link Collections#emptySet()} if the current node is leaf
	 */
	@Override
	public Collection<? extends TreeNode<T>> subtrees() {
		if (isLeaf()) {
			return Collections.emptySet();
		}
		Collection<TreeNode<T>> subtrees = new LinkedHashSet<>();
		subtrees.add(leftMostNode);
		LinkedMultiTreeNode<T> nextSubtree = leftMostNode.rightSiblingNode;
		while (nextSubtree != null) {
			subtrees.add(nextSubtree);
			nextSubtree = nextSubtree.rightSiblingNode;
		}
		return subtrees;
	}

	/**
	 * Adds the subtree with all of its descendants to the current tree node
	 * <p>
	 * {@code null} subtree cannot be added, in this case return result will
	 * be {@code false}
	 * <p>
	 * Checks whether this tree node was changed as a result of the call
	 *
	 * @param subtree subtree to add to the current tree node
	 * @return {@code true} if this tree node was changed as a
	 *         result of the call; {@code false} otherwise
	 */
	@Override
	public boolean add(TreeNode<T> subtree) {
		if (subtree == null) {
			return false;
		}
		linkParent(subtree, this);
		if (isLeaf()) {
			leftMostNode = (LinkedMultiTreeNode<T>) subtree;
			lastSubtreeNode = leftMostNode;
		} else {
			lastSubtreeNode.rightSiblingNode = (LinkedMultiTreeNode<T>) subtree;
			lastSubtreeNode = lastSubtreeNode.rightSiblingNode;
		}
		return true;
	}

	/**
	 * Drops the first occurrence of the specified subtree from the current
	 * tree node
	 * <p>
	 * Checks whether the current tree node was changed as a result of
	 * the call
	 *
	 * @param subtree subtree to drop from the current tree node
	 * @return {@code true} if the current tree node was changed as a result
	 *         of the call; {@code false} otherwise
	 */
	@Override
	public boolean dropSubtree(TreeNode<T> subtree) {
		if (subtree == null
				|| isLeaf()
				|| subtree.isRoot()) {
			return false;
		}
		if (leftMostNode.equals(subtree)) {
			leftMostNode = leftMostNode.rightSiblingNode;
			unlinkParent(subtree);
			((LinkedMultiTreeNode<T>) subtree).rightSiblingNode = null;
			return true;
		} else {
			LinkedMultiTreeNode<T> nextSubtree = leftMostNode;
			while (nextSubtree.rightSiblingNode != null) {
				if (nextSubtree.rightSiblingNode.equals(subtree)) {
					unlinkParent(subtree);
					nextSubtree.rightSiblingNode = nextSubtree.rightSiblingNode.rightSiblingNode;
					((LinkedMultiTreeNode<T>) subtree).rightSiblingNode = null;
					return true;
				} else {
					nextSubtree = nextSubtree.rightSiblingNode;
				}
			}
		}
		return false;
	}

	/**
	 * Removes all the subtrees with all of its descendants from the current
	 * tree node
	 */
	@Override
	public void clear() {
		if (!isLeaf()) {
			LinkedMultiTreeNode<T> nextNode = leftMostNode;
			while (nextNode != null) {
				unlinkParent(nextNode);
				LinkedMultiTreeNode<T> nextNodeRightSiblingNode = nextNode.rightSiblingNode;
				nextNode.rightSiblingNode = null;
				nextNode.lastSubtreeNode = null;
				nextNode = nextNodeRightSiblingNode;
			}
			leftMostNode = null;
		}
	}

	/**
	 * Returns an iterator over the elements in this tree in proper sequence
	 * <p>
	 * The returned iterator is <b>fail-fast</b>
	 *
	 * @return an iterator over the elements in this tree in proper sequence
	 */
	@Override
	public TreeNodeIterator iterator() {
		return new TreeNodeIterator() {

			/**
			 * Returns the leftmost node of the current tree node if the
			 * current tree node is not a leaf
			 *
			 * @return leftmost node of the current tree node if the current
			 *         tree node is not a leaf
			 * @throws TreeNodeException an exception that is thrown in case
			 *                           if the current tree node is a leaf
			 */
			@Override
			protected TreeNode<T> leftMostNode() {
				return leftMostNode;
			}

			/**
			 * Returns the right sibling node of the current tree node if the
			 * current tree node is not root
			 *
			 * @return right sibling node of the current tree node if the current
			 *         tree node is not root
			 * @throws TreeNodeException an exception that may be thrown in case if
			 *                           the current tree node is root
			 */
			@Override
			protected TreeNode<T> rightSiblingNode() {
				return rightSiblingNode;
			}

		};
	}

	/**
	 * Checks whether the current tree node is a leaf, e.g. does not have any
	 * subtrees
	 *
	 * @return {@code true} if the current tree node is a leaf, e.g. does not
	 *         have any subtrees; {@code false} otherwise
	 */
	@Override
	public boolean isLeaf() {
		return leftMostNode == null;
	}

	/**
	 * Checks whether among the current tree node subtrees there is
	 * a specified subtree
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @param subtree subtree whose presence within the current tree
	 *                node children is to be checked
	 * @return {@code true} if among the current tree node subtrees
	 *         there is a specified subtree; {@code false} otherwise
	 */
	@Override
	public boolean hasSubtree(TreeNode<T> subtree) {
		if (subtree == null
				|| isLeaf()
				|| subtree.isRoot()) {
			return false;
		}
		LinkedMultiTreeNode<T> nextSubtree = leftMostNode;
		while (nextSubtree != null) {
			if (nextSubtree.equals(subtree)) {
				return true;
			} else {
				nextSubtree = nextSubtree.rightSiblingNode;
			}
		}
		return false;
	}

	/**
	 * Checks whether the current tree node with all of its descendants
	 * (entire tree) contains the specified node
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @param node node whose presence within the current tree node with
	 *             all of its descendants (entire tree) is to be checked
	 * @return {@code true} if the current node with all of its descendants
	 *         (entire tree) contains the specified node; {@code false}
	 *         otherwise
	 */
	@Override
	public boolean contains(TreeNode<T> node) {
		if (node == null
				|| isLeaf()
				|| node.isRoot()) {
			return false;
		}
		LinkedMultiTreeNode<T> nextSubtree = leftMostNode;
		while (nextSubtree != null) {
			if (nextSubtree.equals(node)) {
				return true;
			}
			if (nextSubtree.contains(node)) {
				return true;
			}
			nextSubtree = nextSubtree.rightSiblingNode;
		}
		return false;
	}

	/**
	 * Removes the first occurrence of the specified node from the entire tree,
	 * starting from the current tree node and traversing in a pre order manner
	 * <p>
	 * Checks whether the current tree node was changed as a result of the call
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @param node node to remove from the entire tree
	 * @return {@code true} if the current tree node was changed as a result of
	 *         the call; {@code false} otherwise
	 */
	@Override
	public boolean remove(TreeNode<T> node) {
		if (node == null
				|| isLeaf()
				|| node.isRoot()) {
			return false;
		}
		if (dropSubtree(node)) {
			return true;
		}
		LinkedMultiTreeNode<T> nextSubtree = leftMostNode;
		while (nextSubtree != null) {
			if (nextSubtree.remove(node)) {
				return true;
			}
			nextSubtree = nextSubtree.rightSiblingNode;
		}
		return false;
	}

	/**
	 * Traverses the tree in a pre ordered manner starting from the
	 * current tree node and performs the traversal action on each
	 * traversed tree node
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @param action action, which is to be performed on each tree
	 *               node, while traversing the tree
	 */
	@Override
	public void traversePreOrder(TraversalAction<TreeNode<T>> action) {
		if (!action.isCompleted()) {
			action.perform(this);
			if (!isLeaf()) {
				LinkedMultiTreeNode<T> nextNode = leftMostNode;
				while (nextNode != null) {
					nextNode.traversePreOrder(action);
					nextNode = nextNode.rightSiblingNode;
				}
			}
		}
	}

	/**
	 * Traverses the tree in a post ordered manner starting from the
	 * current tree node and performs the traversal action on each
	 * traversed tree node
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @param action action, which is to be performed on each tree
	 *               node, while traversing the tree
	 */
	@Override
	public void traversePostOrder(TraversalAction<TreeNode<T>> action) {
		if (!action.isCompleted()) {
			if (!isLeaf()) {
				LinkedMultiTreeNode<T> nextNode = leftMostNode;
				while (nextNode != null) {
					nextNode.traversePostOrder(action);
					nextNode = nextNode.rightSiblingNode;
				}
			}
			action.perform(this);
		}
	}

	/**
	 * Returns the height of the current tree node, e.g. the number of edges
	 * on the longest downward path between that node and a leaf
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @return height of the current tree node, e.g. the number of edges
	 * on the longest downward path between that node and a leaf
	 */
	@Override
	public int height() {
		if (isLeaf()) {
			return 0;
		}
		int height = 0;
		LinkedMultiTreeNode<T> nextNode = leftMostNode;
		while (nextNode != null) {
			height = Math.max(height, nextNode.height());
			nextNode = nextNode.rightSiblingNode;
		}
		return height + 1;
	}

	/**
	 * Returns the collection of nodes, which have the same parent
	 * as the current node; {@link Collections#emptyList()} if the current
	 * tree node is root or if the current tree node has no subtrees
	 * <p>
	 * Overridden to have a faster array implementation
	 *
	 * @return collection of nodes, which have the same parent as
	 *         the current node; {@link Collections#emptyList()} if the
	 *         current tree node is root or if the current tree node has
	 *         no subtrees
	 */
	@Override
	public Collection<? extends MultiTreeNode<T>> siblings() {
		if (isRoot()) {
			String message = String.format("Unable to find the siblings. The tree node %1$s is root", root());
			throw new TreeNodeException(message);
		}
		LinkedMultiTreeNode<T> firstNode = ((LinkedMultiTreeNode<T>) parent()).leftMostNode;
		if (firstNode.rightSiblingNode == null) {
			return Collections.emptySet();
		}
		Collection<MultiTreeNode<T>> siblings = new LinkedHashSet<>();
		LinkedMultiTreeNode<T> nextNode = firstNode;
		while (nextNode != null) {
			if (!nextNode.equals(this)) {
				siblings.add(nextNode);
			}
			nextNode = nextNode.rightSiblingNode;
		}
		return siblings;
	}

}
