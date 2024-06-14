import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from src.authentication import check_password

from streamlit_elements import elements, mui, html

def run():
    if not check_password():
        st.stop()

    log = st.container()

    def write(x):
        log.write(x)

    with elements("dashboard"):
        from streamlit_elements import dashboard
        from streamlit_elements import editor

        layout = [
            # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
            dashboard.Item("first_item", 0, 0, 4, 1),
            dashboard.Item("second_item", 0, 1, 4, 1),
            dashboard.Item("third_item", 0, 2, 4, 1, isResizable=False),
        ]

        def handle_layout_change(current_layout, all_layouts):
            print(current_layout, all_layouts)
        
        def handle_breakpoint_change(new_breakpoint, new_cols):
            print(new_breakpoint, new_cols)
        
        def change(x):
            print(x)

        with dashboard.Grid(layout, onLayoutChange=handle_layout_change, onBreakpointChange=handle_breakpoint_change):
            editor.Monaco(
                height=300,
                defaultValue="Hi everyone!",
                onChange=change,
                key="first_item"
            )
            mui.Paper("Second item", key="second_item")
            mui.Paper("Third item", key="third_item")


if __name__ == "__main__":
    run()