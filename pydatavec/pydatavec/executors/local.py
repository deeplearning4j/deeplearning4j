class LocalExecutor(object):

    def __init__(self):
        pass

    def __call__(self, tp, csv):
        tp = tp.to_java()
        # TODO