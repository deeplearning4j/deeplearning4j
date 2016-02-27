include ../../vars.mk

SUBDIRS := \
cosinesimilarity \
euclidean \
manhattan


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_strided.cu  $(LINK) -o $(shell pwd)/../../output/$@_strided.$(EXTENSION)

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)
