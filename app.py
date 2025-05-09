import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set page config for Streamlit app
st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Terrorism_clean_dataset1.csv", encoding='ISO-8859-1')
    return data

df = load_data()

# Preprocess Data
df.fillna(0, inplace=True)
df['lab_kill'] = df['nkill'].apply(lambda x: 1 if x > 0 else 0)
df['lab_wound'] = df['nwound'].apply(lambda x: 1 if x > 0 else 0)

# Sidebar for Country and Year Range Selection
st.sidebar.header("Filter Data")
selected_country = st.sidebar.selectbox("Select Country", df['country_txt'].unique())
year_range = st.sidebar.slider("Select Year Range", int(df['iyear'].min()), int(df['iyear'].max()), (2000, 2019))

# Filter data based on sidebar selections
country_data = df[(df['country_txt'] == selected_country) & (df['iyear'].between(year_range[0], year_range[1]))]

# Display Data Analysis for Selected Country and Year Range
if not country_data.empty:
    st.title(f"Terrorism Data Analysis for {selected_country}")
    st.write(f"Time Period: {year_range[0]} - {year_range[1]}")

    # Dangerous city analysis
    most_dangerous_city = country_data['city'].value_counts().idxmax() if not country_data['city'].empty else "N/A"
    top_target = country_data['targtype1_txt'].value_counts().idxmax()
    top_group = country_data['gname'].value_counts().idxmax()
    worst_year = country_data['iyear'].value_counts().idxmax()

    st.subheader(f"Most Dangerous City: {most_dangerous_city}")
    st.write(f"Most Targeted Group: {top_target}")
    st.write(f"Most Active Terrorist Group: {top_group}")
    st.write(f"Worst Year by Number of Attacks: {worst_year}")

    # Pie Charts for attack types, targets, weapons, and terrorist groups
    def plot_pie_chart(data, title):
        fig, ax = plt.subplots()
        ax.pie(
            data,
            labels=data.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10},
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
            pctdistance=0.85
        )
        plt.setp(ax.texts, size=8)
        ax.axis('equal')
        plt.title(title, fontsize=16)
        st.pyplot(fig)

    st.subheader("Attack Type Distribution")
    plot_pie_chart(country_data['attacktype1_txt'].value_counts(), "Attack Type Distribution")

    st.subheader("Target Type Distribution")
    plot_pie_chart(country_data['targtype1_txt'].value_counts(), "Target Type Distribution")

    st.subheader("Weapon Type Distribution")
    plot_pie_chart(country_data['weaptype1_txt'].value_counts(), "Weapon Type Distribution")

    st.subheader("Terrorist Group Distribution")
    plot_pie_chart(country_data['gname'].value_counts(), "Terrorist Group Distribution")

# Line Plot for Attack Frequency within Selected Years
st.title("Attack Frequency Over Selected Years")
attack_freq = country_data['iyear'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.plot(attack_freq.index, attack_freq.values, marker='o', color='b')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Attacks")
ax.set_title(f"Number of Attacks in {selected_country} ({year_range[0]}-{year_range[1]})")
st.pyplot(fig)

# Map Visualization for Selected Years
st.title("Attack Map for Selected Years")
m = folium.Map(location=[20, 0], zoom_start=2)
marker_cluster = MarkerCluster().add_to(m)
for _, row in country_data.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"City: {row['city']} | Killed: {row['nkill']} | Wounded: {row['nwound']}",
        icon=folium.Icon(color="red" if row['nkill'] > 0 else "blue")
    ).add_to(marker_cluster)

st_folium(m, width=700, height=500)

# Machine Learning: Success Prediction


# Preparing the dataset for ML
@st.cache_data
def prepare_ml_data(df):
    df_ml = df[["iyear", "imonth", "iday", "success", "attacktype1", "targtype1", "natlty1", "weaptype1", "lab_kill", "lab_wound", "region",
                "latitude", "longitude", "specificity", "vicinity", "extended", "crit1", "suicide"]].dropna()

    df_ml_encoded = pd.get_dummies(df_ml, columns=["attacktype1", "targtype1", "natlty1", "weaptype1", "region"])
    return df_ml_encoded

df_ml_encoded = prepare_ml_data(df)

# Training the XGBoost model
