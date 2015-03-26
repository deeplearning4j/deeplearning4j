/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.berkeley;


public interface Factory<T> {
	T newInstance(Object...args);
	public static class DefaultFactory<T> implements Factory<T> {
		private final Class c;
		public DefaultFactory(Class c) {
      this.c = c;
		}
		public T newInstance(Object... args) {
      try {
        return (T) c.newInstance();
      } catch (Exception e) {
        e.printStackTrace();
      }
      return null;
    }
	}
  
}
