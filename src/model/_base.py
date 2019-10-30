import abc


class ModelBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, X, X2=None):
        """
        Abstract predict that must be implemented in subclass
        """
        pass
