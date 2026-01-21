/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.ops.routing.RoutingPolicy;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.Supplier;

/**
 * Thread-local context for scoped multi-backend execution.
 * Allows setting device preferences and routing policies for a scope.
 *
 * <p>Example usage:</p>
 * <pre>
 * try (MultiBackendContext ctx = new MultiBackendContext.Builder()
 *         .device(gpuDevice)
 *         .build()) {
 *     // All operations in this scope will prefer the GPU
 *     INDArray result = array.mul(2.0);
 * }
 * </pre>
 */
public class MultiBackendContext implements AutoCloseable {

    private static final ThreadLocal<Deque<MultiBackendContext>> contextStack =
            ThreadLocal.withInitial(ArrayDeque::new);

    private final DeviceDescriptor preferredDevice;
    private final RoutingPolicy routingPolicy;
    private final boolean syncOnClose;
    private final boolean autoPin;
    private final MultiBackendContext previousContext;

    private MultiBackendContext(Builder builder) {
        this.preferredDevice = builder.preferredDevice;
        this.routingPolicy = builder.routingPolicy;
        this.syncOnClose = builder.syncOnClose;
        this.autoPin = builder.autoPin;
        this.previousContext = current();

        // Push this context onto the stack
        contextStack.get().push(this);
    }

    /**
     * Get the current context for this thread
     * @return the current context, or null if none
     */
    public static MultiBackendContext current() {
        Deque<MultiBackendContext> stack = contextStack.get();
        return stack.isEmpty() ? null : stack.peek();
    }

    /**
     * Get the preferred device from the current context
     * @return the preferred device, or null if no context or preference
     */
    public static DeviceDescriptor currentDevice() {
        MultiBackendContext ctx = current();
        return ctx != null ? ctx.getPreferredDevice() : null;
    }

    /**
     * Get the preferred device for auto-pinning within the current context.
     * @return the device to auto-pin to, or null if not enabled
     */
    public static DeviceDescriptor currentAutoPinDevice() {
        MultiBackendContext ctx = current();
        return (ctx != null && ctx.isAutoPin()) ? ctx.getPreferredDevice() : null;
    }

    /**
     * Get the routing policy from the current context
     * @return the routing policy, or null if no context or policy
     */
    public static RoutingPolicy currentPolicy() {
        MultiBackendContext ctx = current();
        return ctx != null ? ctx.getRoutingPolicy() : null;
    }

    /**
     * Get the preferred device
     * @return the device, or null if not set
     */
    public DeviceDescriptor getPreferredDevice() {
        return preferredDevice;
    }

    /**
     * Get the routing policy
     * @return the policy, or null if not set
     */
    public RoutingPolicy getRoutingPolicy() {
        return routingPolicy;
    }

    /**
     * Check if operations should sync on close
     * @return true if sync on close
     */
    public boolean isSyncOnClose() {
        return syncOnClose;
    }

    /**
     * Check if new arrays should be auto-pinned to the preferred device.
     * @return true if auto-pin is enabled
     */
    public boolean isAutoPin() {
        return autoPin;
    }

    @Override
    public void close() {
        // Sync if requested
        if (syncOnClose) {
            // Trigger sync through Nd4j if available
            try {
                Nd4j.getExecutioner().commit();
            } catch (Exception e) {
                // Ignore if not available
            }
        }

        // Pop this context from the stack
        Deque<MultiBackendContext> stack = contextStack.get();
        if (!stack.isEmpty() && stack.peek() == this) {
            stack.pop();
        }
    }

    /**
     * Execute a supplier within this context
     * @param supplier the code to execute
     * @param <T> the return type
     * @return the result
     */
    public <T> T execute(Supplier<T> supplier) {
        return supplier.get();
    }

    /**
     * Execute a runnable within this context
     * @param runnable the code to execute
     */
    public void execute(Runnable runnable) {
        runnable.run();
    }

    /**
     * Create a new builder
     * @return builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Execute code with a specific device
     * @param device the device to use
     * @param supplier the code to execute
     * @param <T> the return type
     * @return the result
     */
    public static <T> T withDevice(DeviceDescriptor device, Supplier<T> supplier) {
        try (MultiBackendContext ctx = builder().device(device).autoPin(true).build()) {
            return supplier.get();
        }
    }

    /**
     * Execute code with a specific device
     * @param device the device to use
     * @param runnable the code to execute
     */
    public static void withDevice(DeviceDescriptor device, Runnable runnable) {
        try (MultiBackendContext ctx = builder().device(device).autoPin(true).build()) {
            runnable.run();
        }
    }

    /**
     * Execute code with a specific routing policy
     * @param policy the policy to use
     * @param supplier the code to execute
     * @param <T> the return type
     * @return the result
     */
    public static <T> T withPolicy(RoutingPolicy policy, Supplier<T> supplier) {
        try (MultiBackendContext ctx = builder().policy(policy).build()) {
            return supplier.get();
        }
    }

    /**
     * Execute code with a specific routing policy
     * @param policy the policy to use
     * @param runnable the code to execute
     */
    public static void withPolicy(RoutingPolicy policy, Runnable runnable) {
        try (MultiBackendContext ctx = builder().policy(policy).build()) {
            runnable.run();
        }
    }

    /**
     * Builder for MultiBackendContext
     */
    public static class Builder {
        private DeviceDescriptor preferredDevice;
        private RoutingPolicy routingPolicy;
        private boolean syncOnClose = false;
        private boolean autoPin = false;

        /**
         * Set the preferred device
         * @param device the device
         * @return this builder
         */
        public Builder device(DeviceDescriptor device) {
            this.preferredDevice = device;
            return this;
        }

        /**
         * Set the routing policy
         * @param policy the policy
         * @return this builder
         */
        public Builder policy(RoutingPolicy policy) {
            this.routingPolicy = policy;
            return this;
        }

        /**
         * Enable sync on close
         * @return this builder
         */
        public Builder syncOnClose() {
            this.syncOnClose = true;
            return this;
        }

        /**
         * Set sync on close
         * @param sync whether to sync
         * @return this builder
         */
        public Builder syncOnClose(boolean sync) {
            this.syncOnClose = sync;
            return this;
        }

        /**
         * Enable auto-pinning of new arrays to the preferred device.
         * @return this builder
         */
        public Builder autoPin() {
            return autoPin(true);
        }

        /**
         * Set auto-pinning behavior for new arrays.
         * @param autoPin whether to auto-pin
         * @return this builder
         */
        public Builder autoPin(boolean autoPin) {
            this.autoPin = autoPin;
            return this;
        }

        /**
         * Build the context
         * @return the new context
         */
        public MultiBackendContext build() {
            return new MultiBackendContext(this);
        }
    }
}
