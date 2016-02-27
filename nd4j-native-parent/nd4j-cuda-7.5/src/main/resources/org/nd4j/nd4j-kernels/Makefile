# Every subdirectory with source files must be described here
SUBDIRS := \
src \


export TOP_DIR:= .
export OUTPUT_DIR:= $(TOP_DIR)/output
export SUB_MODULES:= broadcasting indexreduce pairwise_transforms reduce reduce3 scalar transforms 

all: $(SUB_MODULES)
$(SUB_MODULES):
	cd src/$@ && $(MAKE) && cd ../..
default: all

clean: 
	$(shell rm -rf output/*)

install:
    $(shell rm -rf /tmp/kernels && mkdir /tmp/kernels && cp -rf output/* /tmp/kernels)

nd4j-kernels: all
