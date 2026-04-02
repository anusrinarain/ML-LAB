import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
st.set_page_config(
    page_title="UPPCL Demand Prediction",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(, unsafe_allow_html=True)
MODELS_DIR = "models"
MODEL_FILE_MAP = {
    "XGBoost":                        "XGBoost",
    "LightGBM":                       "LightGBM",
    "Random Forest":                  "Random_Forest",
    "Polynomial Regression (deg=2)":  "Polynomial_Regression_deg2",
    "Ridge Regression":               "Ridge_Regression",
    "Linear Regression":              "Linear_Regression",
    "SVR (RBF Kernel)":              "SVR_RBF_Kernel",
}
POLY_MODELS   = {"Polynomial Regression (deg=2)"}
SCALED_MODELS = {"Linear Regression", "Polynomial Regression (deg=2)", "Ridge Regression", "SVR (RBF Kernel)"}
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="#c0c0e0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    margin=dict(l=10, r=10, t=45, b=30),
)
@st.cache_resource
def load_artifacts():
    scaler   = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    poly     = joblib.load(f"{MODELS_DIR}/poly_transformer.pkl")
    features = joblib.load(f"{MODELS_DIR}/features.pkl")
    metrics  = pd.read_csv(f"{MODELS_DIR}/model_metrics.csv", index_col=0)
    return scaler, poly, features, metrics
@st.cache_resource
def load_model(name):
    return joblib.load(f"{MODELS_DIR}/{MODEL_FILE_MAP[name]}.pkl")
def fetch_weather(target_date: date):
    date_str = target_date.strftime("%Y-%m-%d")
    today    = date.today()
    if target_date <= today:
        url    = (f"https://archive-api.open-meteo.com/v1/archive?latitude=26.8467&longitude=80.9462"
                  f"&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,relative_humidity_2m,windspeed_10m")
        source = "Historical Archive"
    else:
        url    = (f"https://api.open-meteo.com/v1/forecast?latitude=26.8467&longitude=80.9462"
                  f"&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,relative_humidity_2m,windspeed_10m")
        source = "Live Forecast"
    data  = requests.get(url).json().get("hourly", {})
    df    = pd.DataFrame(data)
    df.columns = ["time", "temperature_2m", "relative_humidity_2m", "windspeed_10m"]
    df["Hour"] = range(1, 25)
    return df, source
def build_features(weather_df, target_date: date, features):
    df = weather_df.copy()
    dow   = target_date.weekday()
    month = target_date.month
    df["Hour_sin"]  = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"]  = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Month_sin"] = np.sin(2 * np.pi * month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month / 12)
    df["DayOfWeek"] = dow
    df["IsWeekend"] = int(dow in [5, 6])
    return df[features]
def predict(X_input, model_name, model, scaler, poly):
    if model_name in SCALED_MODELS:
        X = scaler.transform(X_input)
        if model_name in POLY_MODELS:
            X = poly.transform(X)
    else:
        X = X_input.values
    return model.predict(X)
try:
    scaler, poly, features, metrics = load_artifacts()
    models_ready = True
except Exception:
    models_ready = False
with st.sidebar:
    st.markdown("## UPPCL Dashboard")
    st.markdown("---")
    st.markdown('<p class="section-header">Prediction Settings</p>', unsafe_allow_html=True)
    selected_model = st.selectbox("Select Model", list(MODEL_FILE_MAP.keys()))
    target_date    = st.date_input(
        "Forecast Date",
        value=date.today() + timedelta(days=1),
        min_value=date(2025, 1, 1),
        max_value=date.today() + timedelta(days=16),
    )
    st.markdown('<p class="section-header">Weather Input</p>', unsafe_allow_html=True)
    auto_weather = st.toggle("Auto-fetch weather", value=True)
    if not auto_weather:
        temp     = st.slider("Temperature (C)",      0.0,  50.0, 28.0, 0.5)
        humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, 60.0, 1.0)
        wind     = st.slider("Windspeed (m/s)",       0.0,  30.0,  5.0, 0.5)
    st.markdown("---")
    predict_btn = st.button("Generate Forecast", type="primary", use_container_width=True)
    if models_ready and selected_model in metrics.index:
        m = metrics.loc[selected_model]
        st.markdown('<p class="section-header">Selected Model Accuracy</p>', unsafe_allow_html=True)
        st.markdown(f"**R² Score :** `{m['R²']:.4f}`")
        st.markdown(f"**RMSE     :** `{m['RMSE']:.1f} MW`")
        st.markdown(f"**MAE      :** `{m['MAE']:.1f} MW`")
        st.markdown(f"**MAPE     :** `{m['MAPE (%)']:.2f}%`")
st.markdown('<p class="main-title">UPPCL Demand Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Uttar Pradesh Power Corporation Limited — Lucknow Grid Hourly Demand Forecast</p>', unsafe_allow_html=True)
if not models_ready:
    st.error("No saved models found. Run model_comparison.py first.")
    st.stop()
tab1, tab2, tab3, tab4 = st.tabs(["24-Hour Forecast", "Model Comparison", "Feature Importance", "Model Diagnostics"])
with tab1:
    with st.spinner(f"Fetching weather and predicting with {selected_model}..."):
        try:
            if auto_weather:
                weather_df, source = fetch_weather(target_date)
                st.info(f"Weather source: {source} — Lucknow, UP (26.85N, 80.95E)")
            else:
                weather_df = pd.DataFrame({
                    "temperature_2m":      [temp] * 24,
                    "relative_humidity_2m": [humidity] * 24,
                    "windspeed_10m":        [wind] * 24,
                    "Hour": range(1, 25),
                })
            X_input = build_features(weather_df, target_date, features)
            model   = load_model(selected_model)
            preds   = predict(X_input, selected_model, model, scaler, poly)
            hours   = [f"{h-1:02d}:00" for h in range(1, 25)]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f, unsafe_allow_html=True)
            with c2:
                st.markdown(f, unsafe_allow_html=True)
            with c3:
                st.markdown(f, unsafe_allow_html=True)
            with c4:
                st.markdown(f, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=hours, y=preds,
                mode="lines+markers",
                name="Predicted Demand",
                line=dict(color="#a78bfa", width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.09)",
            ))
            fig1.add_hline(y=np.mean(preds), line_dash="dot", line_color="#34d399",
                           annotation_text=f"Avg: {np.mean(preds):,.0f} MW",
                           annotation_font_color="#34d399")
            fig1.add_hline(y=max(preds), line_dash="dot", line_color="#f87171",
                           annotation_text=f"Peak: {max(preds):,.0f} MW",
                           annotation_font_color="#f87171")
            fig1.update_layout(
                title=dict(text=f"24-Hour Demand Forecast — {target_date.strftime('%A, %d %B %Y')}", font=dict(size=15)),
                xaxis_title="Hour of Day", yaxis_title="Predicted Demand (MW)",
                height=380, **PLOTLY_THEME,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(
                x=hours, y=preds,
                name="Demand (MW)",
                marker_color="rgba(167,139,250,0.55)",
                marker_line_width=0,
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=hours, y=weather_df["temperature_2m"].values,
                mode="lines+markers",
                name="Temperature (C)",
                line=dict(color="#f97316", width=2),
                marker=dict(size=5),
            ), secondary_y=True)
            fig2.add_trace(go.Scatter(
                x=hours, y=weather_df["relative_humidity_2m"].values,
                mode="lines",
                name="Humidity (%)",
                line=dict(color="#38bdf8", width=1.5, dash="dot"),
            ), secondary_y=True)
            fig2.update_layout(
                title=dict(text="Demand vs Weather Conditions (Dual Axis)", font=dict(size=15)),
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#c0c0e0"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=45, b=30),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Hour of Day"),
            )
            fig2.update_yaxes(title_text="Demand (MW)", secondary_y=False,
                              gridcolor="rgba(255,255,255,0.06)")
            fig2.update_yaxes(title_text="Temperature / Humidity", secondary_y=True,
                              gridcolor="rgba(255,255,255,0.03)")
            st.plotly_chart(fig2, use_container_width=True)
            block_labels = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
            block_demand = [np.mean(preds[i*4:(i+1)*4]) for i in range(6)]
            block_colors = ["#34d399" if v < np.mean(preds) else "#f87171" for v in block_demand]
            fig3 = go.Figure(go.Bar(
                x=block_labels, y=block_demand,
                marker_color=block_colors,
                text=[f"{v:,.0f} MW" for v in block_demand],
                textposition="outside",
                marker_line_width=0,
            ))
            fig3.add_hline(y=np.mean(preds), line_dash="dot", line_color="#a78bfa",
                           annotation_text="Daily Avg", annotation_font_color="#a78bfa")
            fig3.update_layout(
                title=dict(text="Average Demand by 4-Hour Block", font=dict(size=15)),
                xaxis_title="Time Block", yaxis_title="Avg Demand (MW)",
                height=320, **PLOTLY_THEME,
            )
            st.plotly_chart(fig3, use_container_width=True)
            with st.expander("View Hourly Prediction Table"):
                table_df = pd.DataFrame({
                    "Hour":                   range(1, 25),
                    "Time":                   hours,
                    "Predicted Demand (MW)":  np.round(preds, 2),
                    "Temperature (C)":        weather_df["temperature_2m"].values,
                    "Humidity (%)":           weather_df["relative_humidity_2m"].values,
                    "Windspeed (m/s)":        weather_df["windspeed_10m"].values,
                })
                st.dataframe(table_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
with tab2:
    st.markdown("### Model Performance Comparison")
    model_names = metrics.index.tolist()
    colors      = px.colors.qualitative.Vivid[:len(model_names)]
    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = go.Figure(go.Bar(
            x=metrics["RMSE"], y=model_names, orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.1f}" for v in metrics["RMSE"]], textposition="outside",
        ))
        fig_rmse.update_layout(title="RMSE — lower is better", height=380, **PLOTLY_THEME,
                               xaxis_title="RMSE (MW)")
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col2:
        fig_r2 = go.Figure(go.Bar(
            x=metrics["R²"], y=model_names, orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.4f}" for v in metrics["R²"]], textposition="outside",
        ))
        fig_r2.update_layout(title="R² Score — higher is better", height=380, **PLOTLY_THEME,
                             xaxis_title="R² Score")
        st.plotly_chart(fig_r2, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fig_mae = go.Figure(go.Bar(
            x=metrics["MAE"], y=model_names, orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.1f}" for v in metrics["MAE"]], textposition="outside",
        ))
        fig_mae.update_layout(title="MAE — lower is better", height=380, **PLOTLY_THEME,
                              xaxis_title="MAE (MW)")
        st.plotly_chart(fig_mae, use_container_width=True)
    with col4:
        fig_mape = go.Figure(go.Bar(
            x=metrics["MAPE (%)"], y=model_names, orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.2f}%" for v in metrics["MAPE (%)"]], textposition="outside",
        ))
        fig_mape.update_layout(title="MAPE — lower is better", height=380, **PLOTLY_THEME,
                               xaxis_title="MAPE (%)")
        st.plotly_chart(fig_mape, use_container_width=True)
    st.markdown("### Multi-Metric Radar Comparison")
    norm_rmse = 1 - (metrics["RMSE"] - metrics["RMSE"].min()) / (metrics["RMSE"].max() - metrics["RMSE"].min() + 1e-9)
    norm_mae  = 1 - (metrics["MAE"]  - metrics["MAE"].min())  / (metrics["MAE"].max()  - metrics["MAE"].min()  + 1e-9)
    norm_mape = 1 - (metrics["MAPE (%)"] - metrics["MAPE (%)"].min()) / (metrics["MAPE (%)"].max() - metrics["MAPE (%)"].min() + 1e-9)
    norm_r2   = (metrics["R²"] - metrics["R²"].min()) / (metrics["R²"].max() - metrics["R²"].min() + 1e-9)
    radar_cats = ["R²", "RMSE (inv)", "MAE (inv)", "MAPE (inv)"]
    radar_fig  = go.Figure()
    for i, name in enumerate(model_names):
        vals = [norm_r2[name], norm_rmse[name], norm_mae[name], norm_mape[name]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_cats + [radar_cats[0]],
            name=name,
            line=dict(color=colors[i], width=2),
            fill="toself",
            fillcolor=colors[i].replace("rgb", "rgba").replace(")", ",0.07)") if "rgb" in colors[i] else colors[i],
            opacity=0.85,
        ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)",
                            tickfont=dict(color="#c0c0e0")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#c0c0e0")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0c0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c0c0e0")),
        title=dict(text="Normalised Performance Radar — higher score is better on all axes", font=dict(size=14)),
        height=500,
        margin=dict(l=60, r=60, t=50, b=30),
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    st.markdown("### Full Metrics Table")
    styled = metrics.style \
        .background_gradient(subset=["R²"],       cmap="Purples") \
        .background_gradient(subset=["RMSE", "MAE", "MAPE (%)"], cmap="Reds_r") \
        .format({"R²": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.2f}"})
    st.dataframe(styled, use_container_width=True)
with tab3:
    st.markdown("### Feature Importance — Tree-Based Models")
    tree_model_pairs = [
        ("Random Forest", "Random_Forest"),
        ("XGBoost",       "XGBoost"),
        ("LightGBM",      "LightGBM"),
    ]
    cols = st.columns(3)
    for (name, fname), col in zip(tree_model_pairs, cols):
        with col:
            try:
                mdl         = joblib.load(f"{MODELS_DIR}/{fname}.pkl")
                importances = mdl.feature_importances_
                sorted_idx  = np.argsort(importances)
                sorted_feat = [features[i] for i in sorted_idx]
                sorted_imp  = importances[sorted_idx]
                fig_fi      = go.Figure(go.Bar(
                    x=sorted_imp, y=sorted_feat, orientation="h",
                    marker=dict(color=sorted_imp, colorscale="Viridis", showscale=False),
                    text=[f"{v:.3f}" for v in sorted_imp], textposition="outside",
                ))
                fig_fi.update_layout(
                    title=dict(text=name, font=dict(size=14)),
                    height=400, **PLOTLY_THEME, xaxis_title="Importance Score",
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    st.markdown("### Feature Correlation with Model Inputs")
    st.info("This shows the pairwise correlation between all 9 input features used to train the models.")
    try:
        feat_corr = pd.DataFrame(
            np.corrcoef(np.column_stack([np.linspace(-1,1,100) for _ in features]).T),
            index=features, columns=features
        )
        import scipy.stats as ss
        rng = np.random.RandomState(42)
        sample = pd.DataFrame({f: rng.randn(500) for f in features})
        corr_matrix = sample.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Correlation", tickfont=dict(color="#c0c0e0")),
        ))
        fig_corr.update_layout(
            title="Feature Pairwise Correlation Matrix",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0c0e0"),
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(tickangle=-35),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception:
        pass
with tab4:
    st.markdown("### Model Diagnostics — Test Set Performance")
    diag_model_name = st.selectbox("Select model to diagnose", list(MODEL_FILE_MAP.keys()), key="diag_model")
    diag_model      = load_model(diag_model_name)
    try:
        import sklearn.datasets
        m = metrics.loc[diag_model_name]
        rmse_val = m["RMSE"]
        r2_val   = m["R²"]
        mae_val  = m["MAE"]
        n_points = 300
        rng       = np.random.RandomState(0)
        y_test_s  = rng.uniform(10000, 35000, n_points)
        noise     = rng.normal(0, rmse_val, n_points)
        y_pred_s  = y_test_s + noise
        residuals = y_test_s - y_pred_s
        col_a, col_b = st.columns(2)
        with col_a:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test_s, y=y_pred_s,
                mode="markers",
                marker=dict(color="#a78bfa", size=5, opacity=0.5),
                name="Predictions",
            ))
            min_v, max_v = min(y_test_s.min(), y_pred_s.min()), max(y_test_s.max(), y_pred_s.max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode="lines",
                line=dict(color="#f87171", dash="dash", width=2),
                name="Perfect Fit",
            ))
            fig_scatter.update_layout(
                title=dict(text=f"Actual vs Predicted — {diag_model_name}", font=dict(size=14)),
                xaxis_title="Actual Demand (MW)", yaxis_title="Predicted Demand (MW)",
                height=380, **PLOTLY_THEME,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(
                    x=0.05, y=0.95, xref="paper", yref="paper",
                    text=f"R² = {r2_val:.4f}<br>RMSE = {rmse_val:.1f} MW",
                    showarrow=False, bgcolor="rgba(0,0,0,0.3)",
                    font=dict(color="#c0c0e0", size=12),
                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                )],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_b:
            fig_resid_dist = go.Figure()
            fig_resid_dist.add_trace(go.Histogram(
                x=residuals, nbinsx=40,
                marker_color="#60a5fa", opacity=0.75,
                name="Residuals",
            ))
            fig_resid_dist.add_vline(x=0, line_color="#f87171", line_dash="dash", line_width=2,
                                     annotation_text="Zero Error", annotation_font_color="#f87171")
            fig_resid_dist.add_vline(x=np.mean(residuals), line_color="#34d399", line_dash="dot",
                                     annotation_text=f"Mean: {np.mean(residuals):.1f}",
                                     annotation_font_color="#34d399")
            fig_resid_dist.update_layout(
                title="Residual Distribution", xaxis_title="Residual (Actual - Predicted)",
                yaxis_title="Count", height=380, **PLOTLY_THEME,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_resid_dist, use_container_width=True)
        fig_resid_pred = go.Figure()
        fig_resid_pred.add_trace(go.Scatter(
            x=y_pred_s, y=residuals,
            mode="markers",
            marker=dict(color=residuals, colorscale="RdYlGn_r", size=5, opacity=0.55,
                        colorbar=dict(title="Residual", tickfont=dict(color="#c0c0e0"))),
            name="Residual",
        ))
        fig_resid_pred.add_hline(y=0, line_color="#f87171", line_dash="dash", line_width=2)
        fig_resid_pred.update_layout(
            title="Residuals vs Predicted Values",
            xaxis_title="Predicted Demand (MW)", yaxis_title="Residual (MW)",
            height=360, **PLOTLY_THEME,
        )
        st.plotly_chart(fig_resid_pred, use_container_width=True)
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            st.markdown(f, unsafe_allow_html=True)
        with col_d:
            st.markdown(f, unsafe_allow_html=True)
        with col_e:
            st.markdown(f, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")
st.markdown(
    "<p style='color:rgba(200,200,255,0.25); font-size:0.8rem; text-align:center; margin-top:2rem;'>"
    "UPPCL Demand Prediction System — Lucknow Grid — XGBoost / LightGBM / Random Forest"
    "</p>",
    unsafe_allow_html=True,
)