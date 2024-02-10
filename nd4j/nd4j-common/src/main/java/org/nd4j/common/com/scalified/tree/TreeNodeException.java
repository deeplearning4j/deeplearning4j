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

package org.nd4j.common.com.scalified.tree;

/**
 * The class is responsible for different exceptional cases,
 * that may be caused by user actions while working with {@link TreeNode}
 *
 * @author shell
 * @version 1.0.0
 * @since 1.0.0
 */
public class TreeNodeException extends RuntimeException {

	/**
	 * Constructs a new tree node exception with the specified detail message
	 *
	 * @param message the detail message. The detail message is saved for
	 *                later retrieval by the {@link #getMessage()} method
	 */
	public TreeNodeException(String message) {
		super(message);
	}

	/**
	 * Constructs a new tree node exception with the specified detail message and cause
	 *
	 * @param message the detail message. The detail message is saved for
	 *                later retrieval by the {@link #getMessage()} method
	 * @param  cause the cause (which is saved for later retrieval by the
	 *               {@link #getCause()} method). A {@code null} value is
	 *               permitted, and indicates that the cause is nonexistent
	 *               or unknown
	 */
	public TreeNodeException(String message, Throwable cause) {
		super(message, cause);
	}

}
