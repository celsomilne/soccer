from ._base import ModelBase


class SoccerModel(ModelBase):
    def predict(self, frame, prevFrame=None) -> list:
        """Get predicted positions and speeds of players and the ball.
        
        Parameters
        ----------
        frame : np.ndarray
            The frame to apply the predicition to.
        prevFrame : np.ndarray
            The previous frame in the video. If None, `speed` is set to None

        Returns
        -------
        prediction : list of tuples
            Each tuple in `prediction` is (label, (pt1, pt2), speed)
                * label : str
                    The label name
                * pt1 : (int, int)
                    Vertex of the rectangle bounding box as (x_position, y_position)
                * pt2 : (int, int)
                    Vertex of the rectangle opposite to `pt1` as (x_position, y_position)
                * speed : float
                    Speed in m/s of the label. None is `prevFrame` is None.
        """
        return
