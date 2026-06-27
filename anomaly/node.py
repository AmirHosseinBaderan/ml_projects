
class Node:

    def __init__(
        self,
        feature=None,
        split_value=None,
        left=None,
        right=None,
        size=0
    ):
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size