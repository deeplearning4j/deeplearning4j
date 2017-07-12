package org.deeplearning4j.util;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.util.HashUtil;
import org.slf4j.Logger;

import java.util.HashSet;
import java.util.Queue;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.locks.ReentrantLock;

/**
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class OneTimeLogger {
    protected static final Queue<Long> buffer = new LinkedTransferQueue<>();

    private static final ReentrantLock lock = new ReentrantLock();

    protected static boolean isEligible(String message) {
        long hash = HashUtil.getLongHash(message);
        try {
            lock.lock();

            if (buffer.size() >= 100)
                buffer.remove();

            if (buffer.contains(new Long(hash)))
                return false;

            buffer.add(new Long(hash));
            return true;
        } finally {
            lock.unlock();
        }
    }

    public static void info(Logger logger, String format, Object...arguments) {
        if (!isEligible(format))
            return;

        logger.info(format, arguments);
    }

    public static void warn(Logger logger, String format, Object...arguments) {
        if (!isEligible(format))
            return;

        logger.warn(format, arguments);
    }

    public static void error(Logger logger, String format, Object...arguments) {
        if (!isEligible(format))
            return;

        logger.error(format, arguments);
    }

    public static void reset() {
        buffer.clear();
    }
}
