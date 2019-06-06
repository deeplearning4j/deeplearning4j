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

package org.nd4j.linalg.collection;

import org.nd4j.linalg.primitives.Pair;

import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;

/**
 * Created by agibsonccc on 4/29/14.
 */
public class MultiDimensionalSet<K, V> implements Set<Pair<K, V>> {


    private Set<Pair<K, V>> backedSet;

    public MultiDimensionalSet(Set<Pair<K, V>> backedSet) {
        this.backedSet = backedSet;
    }

    public static <K, V> MultiDimensionalSet<K, V> hashSet() {
        return new MultiDimensionalSet<>(new HashSet<Pair<K, V>>());
    }


    public static <K, V> MultiDimensionalSet<K, V> treeSet() {
        return new MultiDimensionalSet<>(new TreeSet<Pair<K, V>>());
    }



    public static <K, V> MultiDimensionalSet<K, V> concurrentSkipListSet() {
        return new MultiDimensionalSet<>(new ConcurrentSkipListSet<Pair<K, V>>());
    }

    /**
     * Returns the number of elements in this applyTransformToDestination (its cardinality).  If this
     * applyTransformToDestination contains more than <tt>Integer.MAX_VALUE</tt> elements, returns
     * <tt>Integer.MAX_VALUE</tt>.
     *
     * @return the number of elements in this applyTransformToDestination (its cardinality)
     */
    @Override
    public int size() {
        return backedSet.size();
    }

    /**
     * Returns <tt>true</tt> if this applyTransformToDestination contains no elements.
     *
     * @return <tt>true</tt> if this applyTransformToDestination contains no elements
     */
    @Override
    public boolean isEmpty() {
        return backedSet.isEmpty();
    }

    /**
     * Returns <tt>true</tt> if this applyTransformToDestination contains the specified element.
     * More formally, returns <tt>true</tt> if and only if this applyTransformToDestination
     * contains an element <tt>e</tt> such that
     * <tt>(o==null&nbsp;?&nbsp;e==null&nbsp;:&nbsp;o.equals(e))</tt>.
     *
     * @param o element whose presence in this applyTransformToDestination is to be tested
     * @return <tt>true</tt> if this applyTransformToDestination contains the specified element
     * @throws ClassCastException   if the type of the specified element
     *                              is incompatible with this applyTransformToDestination
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException if the specified element is null and this
     *                              applyTransformToDestination does not permit null elements
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     */
    @Override
    public boolean contains(Object o) {
        return backedSet.contains(o);
    }

    /**
     * Returns an iterator over the elements in this applyTransformToDestination.  The elements are
     * returned in no particular order (unless this applyTransformToDestination is an instance of some
     * class that provides a guarantee).
     *
     * @return an iterator over the elements in this applyTransformToDestination
     */
    @Override
    public Iterator<Pair<K, V>> iterator() {
        return backedSet.iterator();
    }

    /**
     * Returns an array containing all of the elements in this applyTransformToDestination.
     * If this applyTransformToDestination makes any guarantees as to what order its elements
     * are returned by its iterator, this method must return the
     * elements in the same order.
     * <p/>
     * <p>The returned array will be "safe" in that no references to it
     * are maintained by this applyTransformToDestination.  (In other words, this method must
     * allocate a new array even if this applyTransformToDestination is backed by an array).
     * The caller is thus free to modify the returned array.
     * <p/>
     * <p>This method acts as bridge between array-based and collection-based
     * APIs.
     *
     * @return an array containing all the elements in this applyTransformToDestination
     */
    @Override
    public Object[] toArray() {
        return backedSet.toArray();
    }

    /**
     * Returns an array containing all of the elements in this applyTransformToDestination; the
     * runtime type of the returned array is that of the specified array.
     * If the applyTransformToDestination fits in the specified array, it is returned therein.
     * Otherwise, a new array is allocated with the runtime type of the
     * specified array and the size of this applyTransformToDestination.
     * <p/>
     * <p>If this applyTransformToDestination fits in the specified array with room to spare
     * (i.e., the array has more elements than this applyTransformToDestination), the element in
     * the array immediately following the end of the applyTransformToDestination is applyTransformToDestination to
     * <tt>null</tt>.  (This is useful in determining the length of this
     * applyTransformToDestination <i>only</i> if the caller knows that this applyTransformToDestination does not contain
     * any null elements.)
     * <p/>
     * <p>If this applyTransformToDestination makes any guarantees as to what order its elements
     * are returned by its iterator, this method must return the elements
     * in the same order.
     * <p/>
     * <p>Like the {@link #toArray()} method, this method acts as bridge between
     * array-based and collection-based APIs.  Further, this method allows
     * precise control over the runtime type of the output array, and may,
     * under certain circumstances, be used to save allocation costs.
     * <p/>
     * <p>Suppose <tt>x</tt> is a applyTransformToDestination known to contain only strings.
     * The following code can be used to dump the applyTransformToDestination into a newly allocated
     * array of <tt>String</tt>:
     * <p/>
     * <pre>
     *     String[] y = x.toArray(new String[0]);</pre>
     *
     * Note that <tt>toArray(new Object[0])</tt> is identical in function to
     * <tt>toArray()</tt>.
     *
     * @param a the array into which the elements of this applyTransformToDestination are to be
     *          stored, if it is big enough; otherwise, a new array of the same
     *          runtime type is allocated for this purpose.
     * @return an array containing all the elements in this applyTransformToDestination
     * @throws ArrayStoreException  if the runtime type of the specified array
     *                              is not a supertype of the runtime type of every element in this
     *                              applyTransformToDestination
     * @throws NullPointerException if the specified array is null
     */
    @Override
    public <T> T[] toArray(T[] a) {
        return backedSet.toArray(a);
    }

    /**
     * Adds the specified element to this applyTransformToDestination if it is not already present
     * (optional operation).  More formally, adds the specified element
     * <tt>e</tt> to this applyTransformToDestination if the applyTransformToDestination contains no element <tt>e2</tt>
     * such that
     * <tt>(e==null&nbsp;?&nbsp;e2==null&nbsp;:&nbsp;e.equals(e2))</tt>.
     * If this applyTransformToDestination already contains the element, the call leaves the applyTransformToDestination
     * unchanged and returns <tt>false</tt>.  In combination with the
     * restriction on constructors, this ensures that sets never contain
     * duplicate elements.
     * <p/>
     * <p>The stipulation above does not imply that sets must accept all
     * elements; sets may refuse to add any particular element, including
     * <tt>null</tt>, and throw an exception, as described in the
     * specification for {@link Collection#add Collection.add}.
     * Individual applyTransformToDestination implementations should clearly document any
     * restrictions on the elements that they may contain.
     *
     * @param kvPair element to be added to this applyTransformToDestination
     * @return <tt>true</tt> if this applyTransformToDestination did not already contain the specified
     * element
     * @throws UnsupportedOperationException if the <tt>add</tt> operation
     *                                       is not supported by this applyTransformToDestination
     * @throws ClassCastException            if the class of the specified element
     *                                       prevents it from being added to this applyTransformToDestination
     * @throws NullPointerException          if the specified element is null and this
     *                                       applyTransformToDestination does not permit null elements
     * @throws IllegalArgumentException      if some property of the specified element
     *                                       prevents it from being added to this applyTransformToDestination
     */
    @Override
    public boolean add(Pair<K, V> kvPair) {
        return backedSet.add(kvPair);
    }

    /**
     * Removes the specified element from this applyTransformToDestination if it is present
     * (optional operation).  More formally, removes an element <tt>e</tt>
     * such that
     * <tt>(o==null&nbsp;?&nbsp;e==null&nbsp;:&nbsp;o.equals(e))</tt>, if
     * this applyTransformToDestination contains such an element.  Returns <tt>true</tt> if this applyTransformToDestination
     * contained the element (or equivalently, if this applyTransformToDestination changed as a
     * result of the call).  (This applyTransformToDestination will not contain the element once the
     * call returns.)
     *
     * @param o object to be removed from this applyTransformToDestination, if present
     * @return <tt>true</tt> if this applyTransformToDestination contained the specified element
     * @throws ClassCastException            if the type of the specified element
     *                                       is incompatible with this applyTransformToDestination
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException          if the specified element is null and this
     *                                       applyTransformToDestination does not permit null elements
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws UnsupportedOperationException if the <tt>remove</tt> operation
     *                                       is not supported by this applyTransformToDestination
     */
    @Override
    public boolean remove(Object o) {
        return backedSet.remove(o);
    }

    /**
     * Returns <tt>true</tt> if this applyTransformToDestination contains all of the elements of the
     * specified collection.  If the specified collection is also a applyTransformToDestination, this
     * method returns <tt>true</tt> if it is a <i>subset</i> of this applyTransformToDestination.
     *
     * @param c collection to be checked for containment in this applyTransformToDestination
     * @return <tt>true</tt> if this applyTransformToDestination contains all of the elements of the
     * specified collection
     * @throws ClassCastException   if the types of one or more elements
     *                              in the specified collection are incompatible with this
     *                              applyTransformToDestination
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException if the specified collection contains one
     *                              or more null elements and this applyTransformToDestination does not permit null
     *                              elements
     *                              (<a href="Collection.html#optional-restrictions">optional</a>),
     *                              or if the specified collection is null
     * @see #contains(Object)
     */
    @Override
    public boolean containsAll(Collection<?> c) {
        return backedSet.containsAll(c);
    }

    /**
     * Adds all of the elements in the specified collection to this applyTransformToDestination if
     * they're not already present (optional operation).  If the specified
     * collection is also a applyTransformToDestination, the <tt>addAll</tt> operation effectively
     * modifies this applyTransformToDestination so that its value is the <i>union</i> of the two
     * sets.  The behavior of this operation is undefined if the specified
     * collection is modified while the operation is in progress.
     *
     * @param c collection containing elements to be added to this applyTransformToDestination
     * @return <tt>true</tt> if this applyTransformToDestination changed as a result of the call
     * @throws UnsupportedOperationException if the <tt>addAll</tt> operation
     *                                       is not supported by this applyTransformToDestination
     * @throws ClassCastException            if the class of an element of the
     *                                       specified collection prevents it from being added to this applyTransformToDestination
     * @throws NullPointerException          if the specified collection contains one
     *                                       or more null elements and this applyTransformToDestination does not permit null
     *                                       elements, or if the specified collection is null
     * @throws IllegalArgumentException      if some property of an element of the
     *                                       specified collection prevents it from being added to this applyTransformToDestination
     * @see #add(Object)
     */
    @Override
    public boolean addAll(Collection<? extends Pair<K, V>> c) {
        return backedSet.addAll(c);
    }

    /**
     * Retains only the elements in this applyTransformToDestination that are contained in the
     * specified collection (optional operation).  In other words, removes
     * from this applyTransformToDestination all of its elements that are not contained in the
     * specified collection.  If the specified collection is also a applyTransformToDestination, this
     * operation effectively modifies this applyTransformToDestination so that its value is the
     * <i>intersection</i> of the two sets.
     *
     * @param c collection containing elements to be retained in this applyTransformToDestination
     * @return <tt>true</tt> if this applyTransformToDestination changed as a result of the call
     * @throws UnsupportedOperationException if the <tt>retainAll</tt> operation
     *                                       is not supported by this applyTransformToDestination
     * @throws ClassCastException            if the class of an element of this applyTransformToDestination
     *                                       is incompatible with the specified collection
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException          if this applyTransformToDestination contains a null element and the
     *                                       specified collection does not permit null elements
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>),
     *                                       or if the specified collection is null
     * @see #remove(Object)
     */
    @Override
    public boolean retainAll(Collection<?> c) {
        return backedSet.retainAll(c);
    }

    /**
     * Removes from this applyTransformToDestination all of its elements that are contained in the
     * specified collection (optional operation).  If the specified
     * collection is also a applyTransformToDestination, this operation effectively modifies this
     * applyTransformToDestination so that its value is the <i>asymmetric applyTransformToDestination difference</i> of
     * the two sets.
     *
     * @param c collection containing elements to be removed from this applyTransformToDestination
     * @return <tt>true</tt> if this applyTransformToDestination changed as a result of the call
     * @throws UnsupportedOperationException if the <tt>removeAll</tt> operation
     *                                       is not supported by this applyTransformToDestination
     * @throws ClassCastException            if the class of an element of this applyTransformToDestination
     *                                       is incompatible with the specified collection
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException          if this applyTransformToDestination contains a null element and the
     *                                       specified collection does not permit null elements
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>),
     *                                       or if the specified collection is null
     * @see #remove(Object)
     * @see #contains(Object)
     */
    @Override
    public boolean removeAll(Collection<?> c) {
        return backedSet.removeAll(c);
    }

    /**
     * Removes all of the elements from this applyTransformToDestination (optional operation).
     * The applyTransformToDestination will be empty after this call returns.
     *
     * @throws UnsupportedOperationException if the <tt>clear</tt> method
     *                                       is not supported by this applyTransformToDestination
     */
    @Override
    public void clear() {
        backedSet.clear();
    }



    public boolean contains(K k, V v) {
        return contains(new Pair<>(k, v));
    }

    public void add(K k, V v) {
        add(new Pair<>(k, v));
    }

}
