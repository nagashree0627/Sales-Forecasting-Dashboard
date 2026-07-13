import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    return df


df = load_data()

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Sales Overview",
        "📈 Forecast Explorer",
        "⚠️ Anomaly Report",
        "📦 Product Demand Segments"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Sales Forecasting Project**

    Internship Project

    Developed using:
    - Streamlit
    - Prophet
    - XGBoost
    - SARIMA
    - KMeans
    """
)

# -----------------------------------------------------
# PAGE 1  — Sales Overview Dashboard
# -----------------------------------------------------

if page == "🏠 Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    st.markdown("Explore overall sales performance using interactive KPIs and charts.")

    # ----------------------------
    # Sidebar Filters
    # ----------------------------
    st.sidebar.header("Filters")

    selected_region = st.sidebar.multiselect(
        "Select Region",
        options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique())
    )

    selected_category = st.sidebar.multiselect(
        "Select Category",
        options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique())
    )

    filtered_df = df[
        (df["Region"].isin(selected_region)) &
        (df["Category"].isin(selected_category))
    ]

    
    # KPI Cards
    total_sales = filtered_df["Sales"].sum()
    total_orders = filtered_df["Order ID"].nunique()
    avg_order = filtered_df["Sales"].mean()

    if "Profit" in filtered_df.columns:
        total_profit = filtered_df["Profit"].sum()
    else:
        total_profit = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    col2.metric("🛒 Total Orders", total_orders)
    col3.metric("📦 Avg Order Value", f"${avg_order:,.2f}")
    col4.metric("📈 Total Profit", f"${total_profit:,.2f}")

    st.divider()
    
    # Monthly Sales Trend
    

    st.subheader("📈 Monthly Sales Trend")

    filtered_df["YearMonth"] = filtered_df["Order Date"].dt.to_period("M").astype(str)

    monthly_sales = (
        filtered_df
        .groupby("YearMonth")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="YearMonth",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend",
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True) 

    #Sales by Region & Category
    st.divider()

    col1, col2 = st.columns(2)

    
    # Sales by Region
    

    with col1:

        st.subheader("🌍 Sales by Region")

        region_sales = (
            filtered_df
            .groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig_region = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            color="Region",
            text_auto=".2s",
            title="Total Sales by Region"
        )

        st.plotly_chart(fig_region, use_container_width=True)

    
    # Sales by Category
    

    with col2:

        st.subheader("📦 Sales by Category")

        category_sales = (
            filtered_df
            .groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig_category = px.pie(
            category_sales,
            names="Category",
            values="Sales",
            hole=0.45,
            title="Sales Distribution by Category"
        )

        st.plotly_chart(fig_category, use_container_width=True)

    st.divider()


    # Year-wise Sales Analysis
    st.subheader("📅 Year-wise Sales")

    filtered_df["Year"] = filtered_df["Order Date"].dt.year

    year_sales = (
        filtered_df
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig_year = px.bar(
        year_sales,
        x="Year",
        y="Sales",
        text_auto=".2s",
        color="Year",
        title="Total Sales by Year"
    )

    st.plotly_chart(fig_year, use_container_width=True)

    #Additional Features
    # Top-10 Best Selling Sub-Categories

    st.divider()

    st.subheader("🏆 Top 10 Best Selling Sub-Categories")

    top_products = (
        filtered_df
        .groupby("Sub-Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_top = px.bar(
        top_products,
        x="Sales",
        y="Sub-Category",
        orientation="h",
        color="Sales",
        title="Top 10 Sub-Categories by Sales"
    )

    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig_top, use_container_width=True)

    # Recent Transactions 

    st.divider()

    st.subheader("📋 Recent Sales Records")

    st.dataframe(
        filtered_df.sort_values("Order Date", ascending=False).head(15),
        use_container_width=True
    )
# -----------------------------------------------------
# PAGE 2 — Forecast Explorer
# -----------------------------------------------------
elif page == "📈 Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.markdown(
        "Explore sales forecasts generated by different forecasting models."
    )

    # ------------------------------------------------
    # Load CSV Files
    # ------------------------------------------------

    comparison = pd.read_csv("outputs/model_comparison.csv")
    segment = pd.read_csv("outputs/segment_forecasts.csv")
    prophet = pd.read_csv("outputs/prophet_forecast.csv")
    sarima = pd.read_csv("outputs/sarima_forecast.csv")

    # ------------------------------------------------
    # Model Selection
    # ------------------------------------------------

    model = st.selectbox(
        "Choose Forecast Model",
        comparison["Model"]
    )

    row = comparison[
        comparison["Model"] == model
    ].iloc[0]

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{row['MAE']:.2f}")
    c2.metric("RMSE", f"{row['RMSE']:.2f}")
    c3.metric("MAPE", f"{row['MAPE']:.2f}%")

    st.divider()

    st.subheader("🔮 Forecast Values")

    forecast_table = pd.DataFrame({

        "Month":[
            "Month 1",
            "Month 2",
            "Month 3"
        ],

        "Forecast":[

            row["Forecast Month 1"],
            row["Forecast Month 2"],
            row["Forecast Month 3"]

        ]

    })

    st.dataframe(
        forecast_table,
        use_container_width=True
    )

    #Forecast Chart
    fig = px.line(

        forecast_table,

        x="Month",

        y="Forecast",

        markers=True,

        title=f"{model} Forecast"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    #Segment Forecast
    st.divider()

    st.subheader("📦 Category & Region Forecasts")

    selected_segment = st.selectbox(

        "Select Category / Region",

        segment["Segment"]

    )

    selected = segment[
        segment["Segment"] == selected_segment
    ].iloc[0]

    segment_df = pd.DataFrame({

        "Month":[
            "Month 1",
            "Month 2",
            "Month 3"
        ],

        "Forecast":[

            selected["Forecast Month 1"],
            selected["Forecast Month 2"],
            selected["Forecast Month 3"]

        ]

    })

    st.dataframe(
        segment_df,
        use_container_width=True
    )

    #Segment chart
    fig2 = px.bar(

        segment_df,

        x="Month",

        y="Forecast",

        text_auto=".2s",

        title=f"{selected_segment} Forecast"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    #Growth Indicator
    st.metric(

        "Expected Growth",

        f"{selected['Growth']:.2f}"

    )


# -----------------------------------------------------
# PAGE 3 — Anomaly Report
# -----------------------------------------------------
elif page == "⚠️ Anomaly Report":

    st.title("⚠️ Sales Anomaly Report")

    st.markdown("Identify unusual sales patterns using Isolation Forest and Z-Score Detection.")
    
    anomaly = pd.read_csv("outputs/weekly_anomaly_detection.csv")

    anomaly["Week"] = pd.to_datetime(anomaly["Week"])

    
    # Summary
    
    total_weeks = len(anomaly)

    isolation_count = (anomaly["Isolation"] == "Anomaly").sum()

    zscore_count = anomaly["Z_Anomaly"].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Weeks", total_weeks)

    c2.metric("Isolation Forest Anomalies", isolation_count)

    c3.metric("Z-Score Anomalies", int(zscore_count))

    st.divider()

   
    # Isolation Forest Plot
    

    st.subheader("Isolation Forest Detection")

    fig = px.line(
        anomaly,
        x="Week",
        y="Sales",
        title="Weekly Sales"
    )

    anomaly_points = anomaly[
        anomaly["Isolation"] == "Anomaly"
    ]

    fig.add_scatter(
        x=anomaly_points["Week"],
        y=anomaly_points["Sales"],
        mode="markers",
        name="Isolation Anomaly",
        marker=dict(size=10, color="red", symbol="x")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    
    # Z Score Plot
    
    st.subheader("Z-Score Detection")

    fig2 = px.line(
        anomaly,
        x="Week",
        y="Sales",
        title="Weekly Sales"
    )

    z_points = anomaly[
        anomaly["Z_Anomaly"] == True
    ]

    fig2.add_scatter(
        x=z_points["Week"],
        y=z_points["Sales"],
        mode="markers",
        name="Z-Score Anomaly",
        marker=dict(size=10, color="orange", symbol="diamond")
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    
    # Compare Methods
    
    st.subheader("Comparison")

    st.write(f"Isolation Forest detected **{isolation_count}** anomalies.")

    st.write(f"Z-Score detected **{zscore_count}** anomalies.")

    if isolation_count == zscore_count:
        st.success("Both methods detected the same number of anomalies.")
    else:
        st.info("The methods detected a different number of anomalies because they use different mathematical approaches.")

    st.divider()

    # Anomaly Table
    
    st.subheader("Detected Anomalies")

    detected = anomaly[
        (anomaly["Isolation"] == "Anomaly") |
        (anomaly["Z_Anomaly"] == True)
    ]

    st.dataframe(
        detected,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Anomaly Report",
        detected.to_csv(index=False),
        file_name="Anomaly_Report.csv",
        mime="text/csv"
    )


# -----------------------------------------------------
# PAGE 4
# -----------------------------------------------------

elif page == "📦 Product Demand Segments":

    st.title("📦 Product Demand Segments")

    st.markdown(
        "Product sub-categories are grouped into demand segments using K-Means Clustering."
    )

    cluster = pd.read_csv("outputs/product_segments.csv")

    
    # Summary
    
    total_products = cluster["Sub-Category"].nunique()

    total_clusters = cluster["Cluster"].nunique()

    avg_sales = cluster["TotalSales"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Sub-Categories", total_products)

    c2.metric("Clusters", total_clusters)

    c3.metric("Average Sales", f"${avg_sales:,.2f}")

    st.divider()

    
    # Cluster Scatter Plot

    st.subheader("Demand Clusters")

    fig = px.scatter(
        cluster,
        x="PC1",
        y="PC2",
        color=cluster["Cluster"].astype(str),
        hover_name="Sub-Category",
        size="TotalSales",
        title="K-Means Product Clusters (PCA Projection)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    
    # Cluster Distribution
    

    st.subheader("Products in Each Cluster")

    cluster_count = (
        cluster["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_count.columns = ["Cluster", "Products"]

    fig2 = px.bar(
        cluster_count,
        x="Cluster",
        y="Products",
        text="Products",
        color="Cluster",
        title="Cluster Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    
    # Cluster Details
    
    st.subheader("Cluster Details")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(cluster["Cluster"].unique())
    )

    cluster_data = cluster[
        cluster["Cluster"] == selected_cluster
    ]

    st.dataframe(
        cluster_data,
        use_container_width=True
    )

    st.divider()

    
    # Demand Segment Table
    
    st.subheader("Product Demand Segments")

    st.dataframe(
        cluster[
            [
                "Sub-Category",
                "Segment",
                "Cluster",
                "TotalSales",
                "GrowthRate",
                "Volatility",
                "AverageOrderValue"
            ]
        ],
        use_container_width=True
    )

    st.divider()

    
    # Stocking Strategy
    

    st.subheader("Recommended Stocking Strategy")

    strategy = {
        "High Volume, Stable Demand":
            "Maintain high inventory levels and ensure continuous stock availability.",

        "Growing Demand":
            "Increase inventory gradually to meet rising customer demand.",

        "Low Volume, High Volatility":
            "Maintain limited inventory and monitor sales frequently.",

        "Declining Demand":
            "Reduce inventory levels and avoid overstocking."
    }

    segment_name = cluster_data.iloc[0]["Segment"]

    st.success(f"Selected Segment: {segment_name}")

    if segment_name in strategy:
        st.info(strategy[segment_name])
    else:
        st.info("Monitor inventory based on current sales trends.")

    st.divider()

   
    # Download
    

    st.download_button(
        "📥 Download Product Segments",
        cluster.to_csv(index=False),
        file_name="Product_Segments.csv",
        mime="text/csv"
    )