"""
"""
import torch


class BoundingBox:
    """A rectangular bounding box determines the directed regions."""

    def __init__(self, resolution, box_ratios, margin=0.0):
        """
        Args:
            resolution(int): the resolution of the 2d spatial input
            box_ratios(List[float]):
        Returns:
        """
        assert (
            box_ratios[1] < box_ratios[3]
        ), "the boundary top ratio should be less than bottom"
        assert (
            box_ratios[0] < box_ratios[2]
        ), "the boundary left ratio should be less than right"
        self.left = int((box_ratios[0] - margin) * resolution)
        self.right = int((box_ratios[2] + margin) * resolution)
        self.top = int((box_ratios[1] - margin) * resolution)
        self.bottom = int((box_ratios[3] + margin) * resolution)
        self.height = self.bottom - self.top
        self.width = self.right - self.left
        if self.height == 0:
            self.height = 1
        if self.width == 0:
            self.width = 1

    def sliced_tensor_in_bbox(self, tensor: torch.tensor) -> torch.tensor:
        """ slicing the tensor with bbox area

        Args:
            tensor(torch.tensor): the original tensor in 4d
        Returns:
            (torch.tensor): the reduced tensor inside bbox
        """
        return tensor[:, self.top : self.bottom, self.left : self.right, :]

    def mask_reweight_out_bbox(
        self, tensor: torch.tensor, value: float = 0.0
    ) -> torch.tensor:
        """reweighting value outside bbox

        Args:
            tensor(torch.tensor): the original tensor in 4d
            value(float): reweighting factor default with 0.0
        Returns:
            (torch.tensor): the reweighted tensor
        """
        mask = torch.ones_like(tensor).to(tensor.device) * value
        mask[:, self.top : self.bottom, self.left : self.right, :] = 1
        return tensor * mask

    def mask_reweight_in_bbox(
        self, tensor: torch.tensor, value: float = 0.0
    ) -> torch.tensor:
        """reweighting value within bbox

        Args:
            tensor(torch.tensor): the original tensor in 4d
            value(float): reweighting factor default with 0.0
        Returns:
            (torch.tensor): the reweighted tensor
        """
        mask = torch.ones_like(tensor).to(tensor.device)
        mask[:, self.top : self.bottom, self.left : self.right, :] = value
        return tensor * mask

    def __str__(self):
        """it prints Box(L:%d, R:%d, T:%d, B:%d) for better ingestion"""
        return f"Box(L:{self.left}, R:{self.right}, T:{self.top}, B:{self.bottom})"

    def __rerp__(self):
        """ """
        return f"Box(L:{self.left}, R:{self.right}, T:{self.top}, B:{self.bottom})"


if __name__ == "__main__":
    # Example: second quadrant
    input_res = 32
    left = 0.0
    top = 0.0
    right = 0.5
    bottom = 0.5
    box_ratios = [left, top, right, bottom]
    bbox = BoundingBox(resolution=input_res, box_ratios=box_ratios)

    print(bbox)
    # Box(L:0, R:16, T:0, B:16)
