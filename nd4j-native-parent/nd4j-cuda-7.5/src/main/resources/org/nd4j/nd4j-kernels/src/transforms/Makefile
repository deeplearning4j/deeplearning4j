include ../../vars.mk
INC=-Iinclude/

SUBDIRS := \
abs \
ceil \
cos \
exp \
floor \
log \
neg \
pow \
round \
setrange \
sigmoid \
sign \
sin \
softplus \
sqrt \
tanh \



all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_strided.cu  $(LINK) -o $(shell pwd)/../../output/$@_strided.$(EXTENSION)

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)


