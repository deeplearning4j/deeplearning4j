package org.deeplearning4j.perf.listener;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import oshi.json.SystemInfo;

import java.io.File;
import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class SystemPolling {

    private ScheduledExecutorService scheduledExecutorService;
    private long pollEveryMillis;
    private File outputDirectory;
    private NameProvider nameProvider;
    private ObjectMapper objectMapper = new ObjectMapper(new YAMLFactory());

    private SystemPolling(long pollEveryMillis,File outputDirectory,NameProvider nameProvider) {
        this.pollEveryMillis = pollEveryMillis;
        this.outputDirectory = outputDirectory;
        this.nameProvider = nameProvider;
    }


    public void run() {
        scheduledExecutorService = Executors.newScheduledThreadPool(1);
        scheduledExecutorService.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                SystemInfo systemInfo = new SystemInfo();
                HardwareMetric hardwareMetric = HardwareMetric.fromSystem(systemInfo,nameProvider.nextName());
                File hardwareFile = new File(outputDirectory,hardwareMetric.getName() + ".yml");
                try {
                    objectMapper.writeValue(hardwareFile,hardwareMetric);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        },0,pollEveryMillis, TimeUnit.MILLISECONDS);
    }


    public void stopPolling() {
        scheduledExecutorService.shutdownNow();
    }



    public  interface NameProvider {
        String nextName();
    }

    public static class Builder {
        private long pollEveryMillis;
        private File outputDirectory;

        private NameProvider nameProvider = new NameProvider() {
            @Override
            public String nextName() {
                return UUID.randomUUID().toString();
            }
        };


        public Builder nameProvider(NameProvider nameProvider) {
            this.nameProvider = nameProvider;
            return this;
        }


        public Builder pollEveryMillis(long pollEveryMillis) {
            this.pollEveryMillis = pollEveryMillis;
            return this;
        }

        public Builder outputDirectory(File outputDirectory) {
            this.outputDirectory = outputDirectory;
            return this;
        }

        public SystemPolling build() {
            return new SystemPolling(pollEveryMillis,outputDirectory,nameProvider);
        }

    }

}
