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

package org.arbiter.util;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MultiThreadUtils {
	
	public static ExecutorService newExecutorService() {
		return Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
	}

	public static void parallelTasks(final List<Runnable> tasks, ExecutorService executorService) {
		int tasksCount = tasks.size();
		final CountDownLatch latch = new CountDownLatch(tasksCount);
		for(int i=0;i<tasksCount;i++) {
			final int taskIdx = i;
			executorService.execute(new Runnable() {
				public void run() {
					tasks.get(taskIdx).run();
					latch.countDown();
				}
			});
		}
		
		try {
			latch.await();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
	}
}
