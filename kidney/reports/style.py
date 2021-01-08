from typing import Tuple, Optional


def wide_screen_style(
    max_width: int,
    padding: Tuple[int, int, int, int] = (1, 1, 1, 1),
    color: Optional[str] = None,
    background_color: Optional[str] = None
) -> str:
    left, top, right, bottom = padding
    style = f"""
.reportview-container .main .block-container {{
    max-width: {max_width}px;
    padding-left: {left}rem;
    padding-top: {top}rem;
    padding-right: {right}rem;
    padding-bottom: {bottom}rem;
}}
"""
    if color is not None and background_color is not None:
        style += f"""
.reportview-container .main {{
    color: {color};
    background-color: {background_color};
}}
"""
    return f"<style>{style}</style>"
