import streamlit as st

from kidney.datasets.kaggle import get_reader
from kidney.reports import session, sidebar, read_image
from kidney.reports.auth import with_password
from kidney.reports.style import set_wide_screen
from kidney.reports.widgets.zoom import ZoomController

session_state = session.get(password=False)


@with_password(session_state)
def main():
    set_wide_screen()
    reader = get_reader()
    sample_key, thumb_size = sidebar(reader)
    meta = reader.fetch_meta(sample_key)

    st.header("Image Preview")
    zoom = ZoomController(*read_image(meta, thumb_size))
    zoom.set_zoomed_area()
    zoom.render_selected_area(meta["tiff"], meta.get("mask"))
    zoom.render_zoom_selection(zoom.image, caption=sample_key)


if __name__ == '__main__':
    main()
