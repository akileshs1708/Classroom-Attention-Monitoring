import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def get_css() -> str:
    css_path = Path(__file__).resolve().parent / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            return f"<style>{f.read()}</style>"
    return ""

def render_metric_card(title: str, value: str, trend_class: str = ""):
    st.markdown(f"""
        <div class="glass-card">
            <span class="glass-title">{title}</span>
            <span class="glass-metric {trend_class}">{value}</span>
        </div>
    """, unsafe_allow_html=True)

def render_live_feed(base64_img: str):
    if base64_img:
        st.markdown(f"""
        <div class="glass-card" style="padding: 10px;">
            <span class="glass-title">🔴 LIVE CAMERA FEED</span>
            <img src="data:image/jpeg;base64,{base64_img}" style="width: 100%; border-radius: 8px;">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="height: 480px; display: flex; align-items: center; justify-content: center;">
            <span style="color: #64748b; font-size: 1.2rem;">Camera feed offline</span>
        </div>
        """, unsafe_allow_html=True)

def render_donut_chart(behaviors_dict: dict):
    if not behaviors_dict or "behaviors" not in behaviors_dict or not behaviors_dict["behaviors"]:
        st.markdown("<div class='glass-card' style='height: 300px; text-align: center; color: #64748b;'><br><br>No behaviors detected</div>", unsafe_allow_html=True)
        return
        
    data = behaviors_dict["behaviors"]
    labels = [k.replace("_", " ").title() for k in data.keys()]
    values = list(data.values())
    
    fig = px.pie(
        names=labels, 
        values=values, 
        hole=0.6,
        color_discrete_sequence=["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#cbd5e1",
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.0)
    )
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<span class='glass-title'>Action Recognition</span>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)
    
def render_emotion_chart(emotions_dict: dict):
    if not emotions_dict or "emotions" not in emotions_dict or not emotions_dict["emotions"]:
        st.markdown("<div class='glass-card' style='height: 300px; text-align: center; color: #64748b;'><br><br>No emotions detected</div>", unsafe_allow_html=True)
        return
        
    counts = {"happy": 0, "neutral": 0, "sad": 0, "angry": 0, "surprise": 0}
    for e in emotions_dict["emotions"].values():
        # Real backend schema: {student_name, emotion, confidence}
        emotion = e.get("emotion", "").lower()
        if emotion in counts:
            counts[emotion] += 1
            
    fig = go.Figure(data=[
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=["#10b981", "#94a3b8", "#3b82f6", "#ef4444", "#f59e0b"],
            text=list(counts.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#cbd5e1",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<span class='glass-title'>Aggregated Sentiment</span>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

def render_student_table(df):
    if df.empty:
        st.markdown("<div class='glass-card' style='text-align: center; color: #64748b;'>No students detected.</div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<span class='glass-title'>Live Student Attendance & Attention</span>", unsafe_allow_html=True)
    
    def highlight_attention(val):
        if val == 'HIGH': return 'color: #4ade80; font-weight: bold;'
        if val == 'MEDIUM': return 'color: #facc15; font-weight: bold;'
        if val == 'LOW': return 'color: #f87171; font-weight: bold;'
        return ''
        
    def score_gradient(val):
        color = '#4ade80' if val >= 60 else '#facc15' if val >= 40 else '#f87171'
        return f'color: {color}; font-weight: bold;'
        
    styled_df = df.style.applymap(highlight_attention, subset=['Level']) \
                        .applymap(score_gradient, subset=['Attention %'])
                        
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_timeseries_chart(ts_df: pd.DataFrame):
    """
    Renders a per-student attention score line chart from the timeseries CSV
    written by AttentionScorer.save_reports() every ~1 sec (every 30 frames at 30fps).
    """
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<span class='glass-title'>📈 Attention Score Over Time (per student)</span>", unsafe_allow_html=True)

    if ts_df.empty or "student_name" not in ts_df.columns:
        st.markdown(
            "<p style='color:#64748b; text-align:center; padding: 40px 0;'>"
            "Timeseries data will appear once the pipeline has run for a few seconds…"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ts_df = ts_df.copy()
    if "timestamp" in ts_df.columns:
        import datetime
        ts_df["time"] = pd.to_datetime(ts_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
    else:
        ts_df["time"] = ts_df.get("frame", range(len(ts_df))).astype(str)

    fig = px.line(
        ts_df,
        x="time",
        y="attention_score",
        color="student_name",
        markers=False,
        color_discrete_sequence=["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"],
    )

    fig.add_hrect(y0=60, y1=100, fillcolor="#4ade80", opacity=0.05, line_width=0)
    fig.add_hrect(y0=40, y1=60,  fillcolor="#facc15", opacity=0.05, line_width=0)
    fig.add_hrect(y0=0,  y1=40,  fillcolor="#f87171", opacity=0.05, line_width=0)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            range=[0, 100],
            title="Attention %",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

