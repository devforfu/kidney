from typing import Tuple


def color_to_hex(color: Tuple[int, int, int]) -> str:
    """Converts RGB color tuple into hex string."""

    hex_colors = []
    for channel in color:
        if not (0 <= channel <= 255):
            raise ValueError("color not in range [0; 255]")
        hex_colors.append("{:0>2s}".format(hex(channel).replace("0x", "")))
    return f"#{''.join(hex_colors)}"


def hex_to_color(color: str) -> Tuple[int, int, int]:
    if len(color) != 7:
        raise ValueError(f"invalid hex color: {color}")
    color = color.strip("#")
    r, g, b = color[:2], color[2:4], color[4:6]
    return tuple(int(c, 16) for c in (r, g, b))


if __name__ == '__main__':
    assert color_to_hex((255, 0, 0)) == "#ff0000"
    assert color_to_hex((0, 255, 0)) == "#00ff00"
    assert color_to_hex((0, 0, 255)) == "#0000ff"
    assert hex_to_color("#ff00ff") == color_to_hex((255, 0, 255))
