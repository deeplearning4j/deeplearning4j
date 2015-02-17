package org.deeplearning4j.util;

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
