class Condition(object):

    @property
    def name(self):
        return self.__class__.__name__


class InSet(Condition):
    def __init__(self, column, set):
        self.column = column
        self.set = set


class NotInSet(Condition):
    def __init__(self, column, set):
        self.column = column
        self.set = set


class Equals(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class NotEquals(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class LessThan(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class LessThanOrEqual(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class GreaterThan(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class GreaterThanOrEqual(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value
