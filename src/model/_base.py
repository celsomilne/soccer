import abc

class ModelBase(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def predict(self, X):
        """
        Abstract predict that must be implemented in subclass
        """
        pass