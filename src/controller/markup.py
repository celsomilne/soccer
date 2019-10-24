from . import ModelBase

class MarkupEngine_Base(object):

    def __init__(self):
        return

class MarkupEngine(MarkupEngine_Base):
    
    def __init__(self, model):
        super().__init__()

        # Type check the model, making sure that it is an instance of ModelBase
        if not isinstance(model, ModelBase):
            raise TypeError("model must be of type ModelBase, not %s" % (type(model),))
        self.model = model
        

    def markup(self, frame):
        prediction = self.model.predict(frame)
        return frame