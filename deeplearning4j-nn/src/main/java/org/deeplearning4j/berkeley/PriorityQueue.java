/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.berkeley;

import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * A priority queue based on a binary heap.  Note that this implementation does
 * not efficiently support containment, removal, or element promotion
 * (decreaseKey) -- these methods are therefore not yet implemented.  It is a maximum
 * priority queue, so next() gives the highest-priority object.
 *
 * @author Dan Klein
 */
public class PriorityQueue<E> implements Iterator<E>, Serializable, Cloneable, PriorityQueueInterface<E> {
    private static final long serialVersionUID = 1L;
    int size;
    int capacity;
    List<E> elements;
    double[] priorities;

    protected void grow(int newCapacity) {
        List<E> newElements = new ArrayList<>(newCapacity);
        double[] newPriorities = new double[newCapacity];
        if (size > 0) {
            newElements.addAll(elements);
            System.arraycopy(priorities, 0, newPriorities, 0, priorities.length);
        }
        elements = newElements;
        priorities = newPriorities;
        capacity = newCapacity;
    }

    protected int parent(int loc) {
        return (loc - 1) / 2;
    }

    protected int leftChild(int loc) {
        return 2 * loc + 1;
    }

    protected int rightChild(int loc) {
        return 2 * loc + 2;
    }

    protected void heapifyUp(int loc) {
        if (loc == 0)
            return;
        int parent = parent(loc);
        if (priorities[loc] > priorities[parent]) {
            swap(loc, parent);
            heapifyUp(parent);
        }
    }

    protected void heapifyDown(int loc) {
        int max = loc;
        int leftChild = leftChild(loc);
        if (leftChild < size()) {
            double priority = priorities[loc];
            double leftChildPriority = priorities[leftChild];
            if (leftChildPriority > priority)
                max = leftChild;
            int rightChild = rightChild(loc);
            if (rightChild < size()) {
                double rightChildPriority = priorities[rightChild(loc)];
                if (rightChildPriority > priority && rightChildPriority > leftChildPriority)
                    max = rightChild;
            }
        }
        if (max == loc)
            return;
        swap(loc, max);
        heapifyDown(max);
    }

    protected void swap(int loc1, int loc2) {
        double tempPriority = priorities[loc1];
        E tempElement = elements.get(loc1);
        priorities[loc1] = priorities[loc2];
        elements.set(loc1, elements.get(loc2));
        priorities[loc2] = tempPriority;
        elements.set(loc2, tempElement);
    }

    protected void removeFirst() {
        if (size < 1)
            return;
        swap(0, size - 1);
        size--;
        elements.remove(size);
        heapifyDown(0);
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#hasNext()
     */
    public boolean hasNext() {
        return !isEmpty();
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#next()
     */
    public E next() {
        E first = peek();
        removeFirst();
        return first;
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#remove()
     */
    public void remove() {
        throw new UnsupportedOperationException();
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#peek()
     */
    public E peek() {
        if (size() > 0)
            return elements.get(0);
        throw new NoSuchElementException();
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#getPriority()
     */
    public double getPriority() {
        if (size() > 0)
            return priorities[0];
        throw new NoSuchElementException();
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#size()
     */
    public int size() {
        return size;
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#isEmpty()
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /* (non-Javadoc)
     * @see edu.berkeley.nlp.movingwindow.PriorityQueueInterface#add(E, double)
     */
    public boolean add(E key, double priority) {
        if (size == capacity) {
            grow(2 * capacity + 1);
        }
        elements.add(key);
        priorities[size] = priority;
        heapifyUp(size);
        size++;
        return true;
    }

    public void put(E key, double priority) {
        add(key, priority);
    }

    /**
     * Returns a representation of the queue in decreasing priority order.
     */
    @Override
    public String toString() {
        return toString(size(), false);
    }

    /**
     * Returns a representation of the queue in decreasing priority order,
     * displaying at most maxKeysToPrint elements and optionally printing
     * one element per line.
     *
     * @param maxKeysToPrint maximum number of keys to print
     * @param multiline if is set to true, prints each element on new line. Prints elements in one line otherwise.
     */
    public String toString(int maxKeysToPrint, boolean multiline) {
        PriorityQueue<E> pq = clone();
        StringBuilder sb = new StringBuilder(multiline ? "" : "[");
        int numKeysPrinted = 0;
        NumberFormat f = NumberFormat.getInstance();
        f.setMaximumFractionDigits(5);
        while (numKeysPrinted < maxKeysToPrint && pq.hasNext()) {
            double priority = pq.getPriority();
            E element = pq.next();
            sb.append(element == null ? "null" : element.toString());
            sb.append(" : ");
            sb.append(f.format(priority));
            if (numKeysPrinted < size() - 1)
                sb.append(multiline ? "\n" : ", ");
            numKeysPrinted++;
        }
        if (numKeysPrinted < size())
            sb.append("...");
        if (!multiline)
            sb.append("]");
        return sb.toString();
    }

    /**
     * Returns a counter whose keys are the elements in this priority queue, and
     * whose counts are the priorities in this queue.  In the event there are
     * multiple instances of the same element in the queue, the counter's count
     * will be the sum of the instances' priorities.
     *
     * @return
     */
    public Counter<E> asCounter() {
        PriorityQueue<E> pq = clone();
        Counter<E> counter = new Counter<>();
        while (pq.hasNext()) {
            float priority = (float) pq.getPriority();
            E element = pq.next();
            counter.incrementCount(element, priority);
        }
        return counter;
    }

    /**
     * Returns a clone of this priority queue.  Modifications to one will not
     * affect modifications to the other.
     */
    @Override
    public PriorityQueue<E> clone() {
        PriorityQueue<E> clonePQ = new PriorityQueue<>();
        clonePQ.size = size;
        clonePQ.capacity = capacity;
        clonePQ.elements = new ArrayList<>(capacity);
        clonePQ.priorities = new double[capacity];
        if (size() > 0) {
            clonePQ.elements.addAll(elements);
            System.arraycopy(priorities, 0, clonePQ.priorities, 0, size());
        }
        return clonePQ;
    }

    public PriorityQueue() {
        this(15);
    }

    public PriorityQueue(int capacity) {
        int legalCapacity = 0;
        while (legalCapacity < capacity) {
            legalCapacity = 2 * legalCapacity + 1;
        }
        grow(legalCapacity);
    }

    public static void main(String[] args) {
        PriorityQueue<String> pq = new PriorityQueue<>();
        System.out.println(pq);
        pq.put("one", 1);
        System.out.println(pq);
        pq.put("three", 3);
        System.out.println(pq);
        pq.put("one", 1.1);
        System.out.println(pq);
        pq.put("two", 2);
        System.out.println(pq);
        System.out.println(pq.toString(2, false));
        while (pq.hasNext()) {
            System.out.println(pq.next());
        }
    }
}
