from part import Part
class PartWrapper(Part):
    """
    Wrapper class for Parts. to implement custom __eq__ method.
    """

    def __init__(self, part_id: int, family_id: int):
        super().__init__(part_id, family_id)

    def __eq__(self, other) -> bool:
        return super().equivalent(other)
    
    def __hash__(self) -> int:
        return super().__hash__()