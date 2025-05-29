import streamlit as st


page1 = st.Page("check.py", title='Check Test Data')
page2 = st.Page("models.py", title='Models')
page3 = st.Page("models_2.py", title='Models_2')
# pg = st.navigation([page1, page2])
pg = st.navigation([page2])

pg.run()
