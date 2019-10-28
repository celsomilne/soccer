from ..model import ModelBase

class MarkupEngine_Base(object):

    def __init__(self):
        return

class MarkupEngine(MarkupEngine_Base):
    
    def __init__(self, model):
        super().__init__()
        assert(isinstance(model, type(ModelBase)))
        self.model = model
        

    def markup(self, frame, prevFrame):
        # Prediction should be a list of tuples
        # -> (label, (pt1, pt2), speed)
        prediction = self.model.predict(frame, prevFrame)
        label, (pt1, pt2), speed = prediction
        return frame