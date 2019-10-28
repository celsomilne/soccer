from ._base import ModelBase

class SoccerModel(ModelBase):
    
    def predict(self, frame, prevFrame)  -> list:
        """Get predicted positions and speeds of players and the ball.
        
        Parameters
        ----------
        frame : np.ndarray
            The frame to apply the predicition to.
        prevFrame : np.ndarray
            The previous frame in the video. 

        Returns
        -------
        prediction : list of tuples
            Each tuple in `prediction` is (label, (pt1, pt2), speed)
                * label : str
                    The label name
                * pt1 : int
                    Vertex of the rectangle bounding box
                * pt2 : int
                    Vertex of the rectangle opposite to `pt1`
                * speed : float
                    Speed in m/s of the label
        """
        return