include ../../vars.mk

SUBDIRS := \
add \
div \
equals \
greaterthan \
greaterthanorequal \
lessthan \
lessthanorequal \
max \
min \
mul \
notequals \
rdiv \
rsub \
set \
setvalorless \
sub


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_scalar.cu $(LINK) -o $(shell pwd)/../../output/$@_scalar.$(EXTENSION)

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)

