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
