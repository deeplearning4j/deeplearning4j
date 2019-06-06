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

package org.nd4j.util;

import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;

import java.util.HashSet;
import java.util.Queue;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class OneTimeLogger {
    protected static HashSet<String> hashSet = new HashSet<>();
    protected static final Queue<String> buffer = new LinkedTransferQueue<>();

    private static final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    protected static boolean isEligible(String message) {

        try {
            lock.readLock().lock();

            if (hashSet.contains(message))
                return false;

        } finally {
            lock.readLock().unlock();
        }

        try {
            lock.writeLock().lock();

            if (buffer.size() >= 100) {
                String rem = buffer.remove();
                hashSet.remove(rem);
            }

            buffer.add(message);
            hashSet.add(message);

            return true;
        } finally {
            lock.writeLock().unlock();
        }
    }

    public static void info(Logger logger, String format, Object... arguments) {
        if (!isEligible(format))
            return;

        logger.info(format, arguments);
    }

    public static void warn(Logger logger, String format, Object... arguments) {
        if (!isEligible(format))
            return;

        logger.warn(format, arguments);
    }

    public static void error(Logger logger, String format, Object... arguments) {
        if (!isEligible(format))
            return;

        logger.error(format, arguments);
    }

    public static void reset() {
        buffer.clear();
    }
}
