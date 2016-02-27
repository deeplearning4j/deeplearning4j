include ../../vars.mk
SUBDIRS := \
add \
div \
mul \
rdiv \
rsub \
softmax \
sub \
eps \
copy \
eq \
gt \
lt \

all: $(SUBDIRS)
$(SUBDIRS):
	echo "In $@"
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_strided.cu $(LINK)  -o $(shell pwd)/../../output/$@_strided.$(EXTENSION)

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)

