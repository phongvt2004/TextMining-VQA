import streamlit as st


page1 = st.Page("check.py", title='Check Test Data')
page2 = st.Page("models.py", title='Models')
pg = st.navigation([page1, page2])

pg.run()
