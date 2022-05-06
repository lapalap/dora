
class Dora:
    types = {}

    def __class_getitem__(cls, key):
        try:
            W1, W2 = key
        except ValueError:
            raise Exception('ProductWeight[] takes exactly two arguments.')

        name = f'{ProductWeight.__name__}<{W1.__name__}, {W2.__name__}>'

        try:
            return cls.types[name]
        except KeyError:
            pass

        new_type = type(name, (), {'__init__': cls.init})
        new_type.W1 = W1
        new_type.W2 = W2
        new_type.zero = classmethod(cls.zero)
        cls.types[name] = new_type

        return new_type

    def __init__(self):
        raise Exception('ProductWeight is a static class and cannot be instantiated.')

    def init(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

    def zero(cls):
        return cls(cls.W1.zero(), cls.W2.zero())
