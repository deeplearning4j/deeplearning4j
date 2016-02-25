include ../../vars.mk
CC:=nvcc
SUBDIRS := \
broadcastadd \
broadcastcopy \
broadcastdiv \
broadcastmul \
broadcastrdiv \
broadcastrsub \
broadcastsub


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@.cu  $(LINK) -o $(shell pwd)/../../output/$@.$(EXTENSION)

.PHONY: all $(SUBDIRS)

