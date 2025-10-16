"""Streamlit dashboard for Happinometer and HR analytics."""

from __future__ import annotations

import io
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from plotly import express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 HR Analytics Dashboard")
st.caption(
    "อัปโหลดไฟล์ Excel (2568) เพื่อสำรวจความผูกพัน ความสุข และความพึงพอใจของบุคลากร"
)


def _clean_sheet(xls: pd.ExcelFile, sheet_name: str, skiprows: int) -> pd.DataFrame:
    df = xls.parse(sheet_name, skiprows=skiprows)
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df.columns = [str(col).strip() for col in df.columns]
    return df


@st.cache_data(show_spinner=True)
def load_excel(file: io.BytesIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(file)

    commitment_df = _clean_sheet(xls, "ความผูกผัน 68", skiprows=2)
    satisfaction_df = _clean_sheet(xls, "ความพึงพอใจ 68", skiprows=2)
    happiness_df = _clean_sheet(xls, "Happinoter 68", skiprows=2)
    resignation_df = _clean_sheet(xls, "บุคลากรลาออก 68", skiprows=1)

    # Rename main analytical columns for easier reference
    commitment_cols = list(commitment_df.columns)
    rename_commitment = {
        commitment_cols[0]: "No",
        commitment_cols[1]: "Unit",
        commitment_cols[2]: "Respondents",
        commitment_cols[3]: "Mean",
        commitment_cols[4]: "SD",
        commitment_cols[5]: "Percent",
    }
    commitment_df = commitment_df.rename(columns=rename_commitment)

    satisfaction_cols = list(satisfaction_df.columns)
    satisfaction_labels = [
        "No",
        "Unit",
        "Responses",
        "Command_Mean",
        "Command_SD",
        "Command_Percent",
        "Command_Interp",
        "Job_Mean",
        "Job_SD",
        "Job_Percent",
        "Job_Interp",
        "Progress_Mean",
        "Progress_SD",
        "Progress_Percent",
        "Progress_Interp",
        "Environment_Mean",
        "Environment_SD",
        "Environment_Percent",
        "Environment_Interp",
        "Relationship_Mean",
        "Relationship_SD",
        "Relationship_Percent",
        "Relationship_Interp",
        "Compensation_Mean",
        "Compensation_SD",
        "Compensation_Percent",
        "Compensation_Interp",
        "Overall_Mean",
        "Overall_SD",
        "Overall_Percent",
        "Overall_Interp",
    ]
    for idx, label in enumerate(satisfaction_labels):
        if idx < len(satisfaction_cols):
            satisfaction_cols[idx] = label
    satisfaction_df.columns = satisfaction_cols

    happiness_cols = list(happiness_df.columns)
    happiness_labels = [
        "Unit",
        "Body_Mean",
        "Body_Percent",
        "Relax_Mean",
        "Relax_Percent",
        "Heart_Mean",
        "Heart_Percent",
        "Soul_Mean",
        "Soul_Percent",
        "Family_Mean",
        "Family_Percent",
        "Society_Mean",
        "Society_Percent",
        "Brain_Mean",
        "Brain_Percent",
        "Money_Mean",
        "Money_Percent",
        "HappyPlus_Mean",
        "HappyPlus_Percent",
        "Engagement_Mean",
        "Engagement_Percent",
        "WorkLife_Mean",
        "WorkLife_Percent",
        "Say_Mean",
        "Say_Percent",
        "Stay_Mean",
        "Stay_Percent",
        "Strive_Mean",
        "Strive_Percent",
        "OverallHappy_Mean",
        "OverallHappy_Percent",
    ]
    for idx, label in enumerate(happiness_labels):
        if idx < len(happiness_cols):
            happiness_cols[idx] = label
    happiness_df.columns = happiness_cols

    resignation_cols = list(resignation_df.columns)
    rename_resignation = {
        resignation_cols[0]: "Unit",
        resignation_cols[1]: "N",
        resignation_cols[2]: "PN",
        resignation_cols[3]: "Other",
        resignation_cols[4]: "Admin",
        resignation_cols[5]: "Total",
    }
    resignation_df = resignation_df.rename(columns=rename_resignation)

    return commitment_df, satisfaction_df, happiness_df, resignation_df


uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel", type=["xlsx"])

if uploaded_file:
    commitment_df, satisfaction_df, happiness_df, resignation_df = load_excel(uploaded_file)

    st.success("โหลดข้อมูลสำเร็จ! เลือกแท็บต่าง ๆ เพื่อสำรวจเชิงลึก")

    tab_overview, tab_satisfaction, tab_model, tab_happy = st.tabs(
        [
            "ภาพรวม",
            "ความพึงพอใจ",
            "การลาออก & โมเดล",
            "โปรไฟล์ความสุข",
        ]
    )

    with tab_overview:
        st.subheader("ความผูกพันของบุคลากร")
        st.dataframe(
            commitment_df[["Unit", "Mean", "SD", "Percent"]].style.format(
                {"Mean": "{:.2f}", "SD": "{:.2f}", "Percent": "{:.2f}"}
            ),
            use_container_width=True,
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.metric("คะแนนผูกพันเฉลี่ย", f"{commitment_df['Percent'].mean():.2f}%")
        with col2:
            top_unit = commitment_df.sort_values("Percent", ascending=False).iloc[0]
            st.metric("หน่วยคะแนนสูงสุด", f"{top_unit['Unit']} ({top_unit['Percent']:.2f}%)")

    with tab_satisfaction:
        st.subheader("สถิติพรรณนาความพึงพอใจ")
        st.dataframe(
            satisfaction_df.describe(include=np.number).style.format("{:.2f}"),
            use_container_width=True,
        )

        numeric_cols = satisfaction_df.select_dtypes(include=np.number)
        if numeric_cols.shape[1] >= 2:
            st.subheader("Heatmap ความสัมพันธ์")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap="flare", ax=ax)
            st.pyplot(fig)
        else:
            st.info("ไม่พบข้อมูลตัวเลขเพียงพอสำหรับคำนวณค่าสหสัมพันธ์")

    with tab_model:
        st.subheader("โมเดลพยากรณ์การลาออก (OLS)")
        merged_df = pd.merge(
            satisfaction_df[["Unit", "Command_Mean", "Job_Mean", "Overall_Mean"]],
            resignation_df[["Unit", "Total"]],
            on="Unit",
            how="inner",
        ).dropna()

        if not merged_df.empty:
            X = merged_df[["Command_Mean", "Job_Mean", "Overall_Mean"]]
            y = merged_df["Total"]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            st.text(model.summary())
        else:
            st.warning("ไม่พบข้อมูลที่เชื่อมโยงกันระหว่างความพึงพอใจและการลาออก")

        st.divider()
        st.subheader("ภาพรวมการลาออก")
        st.dataframe(resignation_df, use_container_width=True)

    with tab_happy:
        st.subheader("กลุ่มโปรไฟล์ความสุข (K-Means)")
        happy_numeric = happiness_df.select_dtypes(include=np.number).dropna()

        if happy_numeric.shape[0] >= 3:
            scaler = StandardScaler()
            happy_scaled = scaler.fit_transform(happy_numeric)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(happy_scaled)
            happiness_df = happiness_df.assign(Cluster=clusters)

            fig = px.scatter(
                happiness_df,
                x="Body_Mean",
                y="Relax_Mean",
                color="Cluster",
                hover_data=["Unit"],
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(happiness_df[["Unit", "Cluster"]], use_container_width=True)
        else:
            st.warning("ข้อมูลไม่เพียงพอสำหรับการจัดกลุ่ม")
else:
    st.info(
        "กรุณาอัปโหลดไฟล์ Excel ที่มีแผ่นงานตามรายงานปี 2568 เพื่อเริ่มต้นวิเคราะห์"
    )
