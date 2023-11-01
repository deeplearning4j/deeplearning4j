package org.eclipse.deeplearning4j.tests.extensions;

import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.InvocationInterceptor;
import org.junit.jupiter.api.extension.ReflectiveInvocationContext;
import org.junit.jupiter.api.extension.TestWatcher;

/** For ordered tests only, fail fast. */
public class FailFast implements InvocationInterceptor, TestWatcher {
    private static final Map<Integer, Boolean> CLASS_FAILED = new HashMap<>(Map.of(0, false));
    private final Map<Integer, Boolean> methodSucceeded = new HashMap<>(Map.of(0, true));

    @Override
    public void interceptTestMethod(
            Invocation<Void> invocation,
            ReflectiveInvocationContext<Method> invocationContext,
            ExtensionContext extensionContext)
            throws Throwable {
        var classOrder = extensionContext.getRequiredTestClass().getAnnotation(Order.class);
        if (classOrder != null) assumeFalse(CLASS_FAILED.getOrDefault(classOrder.value() - 1, false));
        var methodOrder = extensionContext.getRequiredTestMethod().getAnnotation(Order.class);
        if (methodOrder != null)
            assumeTrue(methodSucceeded.getOrDefault(methodOrder.value() - 1, false));
        invocation.proceed();
    }

    @Override
    public void testSuccessful(ExtensionContext context) {
        var methodOrder = context.getRequiredTestMethod().getAnnotation(Order.class);
        if (methodOrder != null) methodSucceeded.put(methodOrder.value(), true);
    }

    @Override
    public void testFailed(ExtensionContext context, Throwable cause) {
        var classOrder = context.getRequiredTestClass().getAnnotation(Order.class);
        if (classOrder != null) CLASS_FAILED.put(classOrder.value(), true);
    }
}