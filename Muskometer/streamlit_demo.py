import streamlit as st
import numpy as np
import pandas as pd
import time
import altair as alt
from vega_datasets import data

st.title('Muskometer')
st.write('')
st.write('Reading the mind of Elon Musk for fun and profit!')

tsla_df = pd.read_csv('../data/raw/tsla_stock_price.csv')\
                        .drop('Unnamed: 0',axis='columns')
tsla_df['DateTime'] = pd.to_datetime(tsla_df['DateTime'])


line_t = alt.Chart(tsla_df).mark_line(
    color='black',
    size=3
).transform_window(
    rolling_mean='mean(Open)',
    frame=[-15, 15]
).encode(
    x='DateTime:T',
    y='rolling_mean:Q'
)

points_t = alt.Chart(tsla_df).mark_point().encode(
    x='DateTime:T',
    y=alt.Y('Close:Q',
            axis=alt.Axis(title='Price'))
)

st.altair_chart(points_t.interactive() + line_t.interactive())




###### Vega stuff for a demo
source = data.seattle_weather()
source.dtypes
source
line = alt.Chart(source).mark_line(
    color='red',
    size=3
).transform_window(
    rolling_mean='mean(temp_max)',
    frame=[-15, 15]
).encode(
    x='date:T',
    y='rolling_mean:Q'
)

points = alt.Chart(source).mark_point().encode(
    x='date:T',
    y=alt.Y('temp_max:Q',
            axis=alt.Axis(title='Max Temp'))
)

points.interactive() + line.interactive()
###################
## Generate some random data
#df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#
## Build a scatter chart using altair. I modified the example at
## https://altair-viz.github.io/gallery/scatter_tooltips.html
#scatter_chart = st.altair_chart(
#    alt.Chart(df)
#        .mark_circle(size=60)
#        .encode(x='a', y='b', color='c')
#        .interactive()
#)
#
## Append more random data to the chart using add_rows
#for ii in range(0, 100):
#    df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#    scatter_chart.add_rows(df)
#    # Sleep for a moment just for demonstration purposes, so that the new data
#    # animates in.
#    time.sleep(0.1)