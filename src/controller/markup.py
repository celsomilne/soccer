from ..model import ModelBase

class MarkupEngine_Base(object):

    def __init__(self):
        return

class MarkupEngine(MarkupEngine_Base):
    
    def __init__(self, model):
        super().__init__()
        assert(isinstance(model, type(ModelBase)))
        self.model = model
        

    def markup(self, frame):
        prediction = self.model.predict(frame)
        # TODO - mark up the frame
        return frame