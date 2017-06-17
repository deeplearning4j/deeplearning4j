/**
 *
 * APDPlat - Application Product Development Platform
 * Copyright (c) 2013, 杨尚川, yang-shangchuan@qq.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package org.apdplat.word.util;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * 因为Java没有提供AtomicFloat
 * 所以自己实现一个
 * @author 杨尚川
 */
public class AtomicFloat extends Number {

    private AtomicInteger bits;

    public AtomicFloat() {
        this(0f);
    }

    public AtomicFloat(float initialValue) {
        bits = new AtomicInteger(Float.floatToIntBits(initialValue));
    }

    public final float addAndGet(float delta){
        float expect;
        float update;
        do {
            expect = get();
            update = expect + delta;
        } while(!this.compareAndSet(expect, update));

        return update;
    }

    public final float getAndAdd(float delta){
        float expect;
        float update;
        do {
            expect = get();
            update = expect + delta;
        } while(!this.compareAndSet(expect, update));

        return expect;
    }

    public final float getAndDecrement(){
        return getAndAdd(-1);
    }

    public final float decrementAndGet(){
        return addAndGet(-1);
    }

    public final float getAndIncrement(){
        return getAndAdd(1);
    }

    public final float incrementAndGet(){
        return addAndGet(1);
    }

    public final float getAndSet(float newValue) {
        float expect;
        do {
            expect = get();
        } while(!this.compareAndSet(expect, newValue));

        return expect;
    }

    public final boolean compareAndSet(float expect, float update) {
        return bits.compareAndSet(Float.floatToIntBits(expect), Float.floatToIntBits(update));
    }

    public final void set(float newValue) {
        bits.set(Float.floatToIntBits(newValue));
    }

    public final float get() {
        return Float.intBitsToFloat(bits.get());
    }

    public float floatValue() {
        return get();
    }

    public double doubleValue() {
        return (double) floatValue();
    }

    public int intValue() {
        return (int) get();
    }

    public long longValue() {
        return (long) get();
    }

    public String toString() {
        return Float.toString(get());
    }
}
