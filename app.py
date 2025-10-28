# -*- coding: utf-8 -*-
# 최근 L년 평균 vs 실제 — 연속구간 비교 + "월별 vs 연평균" R² 비교
# 백테스트 요약: 1–4년 vs 5–8년만 표시

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="최근 L년 평균 vs 실제 — 연속구간 + 월별 vs 연평균 비교", layout="wide")

# ---------- 레이아웃 ----------
def tidy_layout(fig, title=None, height=360):
    if title:
        fig.update_layout(title=title, title_pad=dict(t=36, l=6, r=6, b=6))
    fig.update_layout(height=height, margin=dict(l=70, r=30, t=90, b=70))
    return fig

# ---------- 파서(일→월 평균 집계 지원) ----------
MONTH_ALIASES = {
    "jan":1,"january":1,"1":1,"01":1,"1월":1,
    "feb":2,"february":2,"2":2,"02":2,"2월":2,
    "mar":3,"march":3,"3":3,"03":3,"3월":3,
    "apr":4,"april":4,"4":4,"04":4,"4월":4,
    "may":5,"5":5,"05":5,"5월":5,
    "jun":6,"june":6,"6":6,"06":6,"6월":6,
    "jul":7,"july":7,"7":7,"07":7,"7월":7,
    "aug":8,"august":8,"8":8,"08":8,"8월":8,
    "sep":9,"sept":9,"september":9,"9":9,"09":9,"9월":9,
    "oct":10,"october":10,"10":10,"10월":10,
    "nov":11,"november":11,"11":11,"11월":11,
    "dec":12,"december":12,"12":12,"12월":12,
}
def norm_month(col)->int|None:
    s = str(col).strip().lower().replace(" ", "")
    s = s.replace("month","").replace("월평균","").replace("평균","")
    if re.fullmatch(r"\d+\s*월", str(col)):
        s = str(col).strip().lower().replace(" ","").replace("월","")
    return MONTH_ALIASES.get(s)

def _clean(s: str) -> str:
    return re.sub(r"[()\[\]{}℃°/·\s]", "", str(s)).lower()

def try_parse_wide(df_raw: pd.DataFrame):
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy()
        body.columns = hdr
        body = body.rename(columns={body.columns[0]: "year"})
        month_map = {}
        for c in body.columns[1:]:
            m = norm_month(c)
            if m: month_map[c]=m
        if len(month_map) >= 6:
            use = ["year"] + list(month_map.keys())
            body = body[use].rename(columns=month_map)
            body["year"] = pd.to_numeric(body["year"], errors="coerce")
            body = body.dropna(subset=["year"])
            body["year"] = body["year"].astype(int)
            long = body.melt(id_vars="year", var_name="month", value_name="temp")
            long["month"] = long["month"].astype(int)
            long["temp"]  = pd.to_numeric(long["temp"], errors="coerce")
            long = long.dropna(subset=["temp"]).sort_values(["year","month"])
            if long["year"].nunique() >= 6:
                return long
    return None

def try_parse_long(df_raw: pd.DataFrame):
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy(); body.columns = hdr

        date_col = None
        for c in body.columns:
            if _clean(c) in ["날짜","date","일자","일시","dt"]:
                date_col = c; break
        if date_col is None:
            for c in body.columns:
                s = pd.to_datetime(body[c], errors="coerce")
                if s.notna().sum() >= max(12, int(len(s)*0.3)):
                    date_col = c; break
        if date_col is None: 
            continue

        val_col = None
        for c in body.columns:
            norm = _clean(c)
            if norm.startswith("평균기온") or norm in ["평균기온","기온","temp","temperature"]:
                val_col = c; break
        if val_col is None:
            nums = [c for c in body.columns if c!=date_col and pd.to_numeric(body[c], errors="coerce").notna().sum()>=max(12, int(len(body)*0.3))]
            if nums: val_col = nums[0]
        if val_col is None: 
            continue

        df = body[[date_col, val_col]].copy().rename(columns={date_col:"date", val_col:"temp"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df = df.dropna(subset=["date","temp"])
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month
        long = (df.groupby(["year","month"], as_index=False)["temp"]
                  .mean()
                  .sort_values(["year","month"]))
        if long["year"].nunique() >= 6:
            return long
    return None

@st.cache_data
def load_excel_any(path_or_buf):
    xls = pd.ExcelFile(path_or_buf)
    for sh in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sh, header=None)
        parsed = try_parse_wide(raw)
        if parsed is not None: return parsed, sh, "wide"
        parsed = try_parse_long(raw)
        if parsed is not None: return parsed, sh, "long"
    return None, None, None

# ---------- 지표 ----------
def r2(y, yp):
    y = np.array(y, dtype=float); yp = np.array(yp, dtype=float)
    m = ~(np.isnan(y) | np.isnan(yp)); y, yp = y[m], yp[m]
    if y.size < 2: return np.nan
    sse = np.sum((y-yp)**2); sst = np.sum((y-y.mean())**2)
    return float(1 - sse/sst) if sst>0 else np.nan

def mae(y, yp):
    y = np.array(y, dtype=float); yp = np.array(yp, dtype=float)
    m = ~(np.isnan(y) | np.isnan(yp)); y, yp = y[m], yp[m]
    return float(np.mean(np.abs(y-yp))) if y.size>0 else np.nan

# ---------- 예측 ----------
def build_y_true(df, Y:int):
    y_df = df.query("year == @Y").sort_values("month")
    return y_df["month"].tolist(), y_df["temp"].to_numpy()

def build_pred_monthly(df, start:int, end:int, months_order:list):
    train = df.query("year >= @start and year <= @end").copy()
    preds=[]
    for m in months_order:
        x = train.loc[train["month"]==m, "temp"].to_numpy()
        preds.append(np.mean(x) if x.size>0 else np.nan)
    return np.array(preds, dtype=float)

def build_pred_annual(df, start:int, end:int, months_order:list):
    g = df.query("year >= @start and year <= @end").groupby("year")["temp"].mean()
    if g.size == 0: return np.full(len(months_order), np.nan)
    scalar = float(g.mean())
    return np.full(len(months_order), scalar, dtype=float)

# ---------- 데이터 로딩 ----------
default_path = Path("기온_198001_202509.xlsx")
uploaded = st.file_uploader("기온 파일(.xlsx) — [연도|1..12] 또는 [날짜|평균기온(℃)]", type=["xlsx"])
df, used_sheet, mode = (load_excel_any(uploaded) if uploaded else
                        (load_excel_any(default_path) if default_path.exists() else (None,None,None)))
if df is None:
    st.error("월별 평균기온을 찾지 못했어. 형식: A)[연도|1..12] 또는 B)[날짜|평균기온(℃)] — 일 데이터는 월 평균으로 자동 집계함.")
    st.stop()

years = sorted(df["year"].unique())
min_year, max_year = int(min(years)), int(max(years))

# ================== 탭 ==================
tab1, tab2 = st.tabs(["단일연도 검증(월별 vs 연평균 비교)", "백테스트 요약"])

# ---------- 탭1 ----------
with tab1:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        target_year = st.number_input("대상연도(실제값 존재)", min_value=min_year+1, max_value=max_year, value=max_year)
    with c2:
        Lmax = st.slider("비교할 최대 L(년)", 3, 15, 10)
    with c3:
        show_table = st.toggle("결과 테이블 표시", value=True)

    st.caption(
        f"평가 규칙(연속 구간): 대상연도 {target_year}일 때 L=1→[{target_year-1}], "
        f"L=2→[{target_year-2}~{target_year-1}], … L=k→[Y-k~Y-1]. "
        "월별 방식은 동월 평균(계절성 유지), 연평균 방식은 L년 연평균의 평균값을 12개월 동일 적용(계절성 제거)."
    )

    months, y_true = build_y_true(df, target_year)

    rows=[]
    for L in range(1, Lmax+1):
        start = target_year - L
        if start < min_year: continue
        y_pred_m = build_pred_monthly(df, start, target_year-1, months)
        y_pred_a = build_pred_annual(df, start, target_year-1, months)
        rows.append((L, r2(y_true, y_pred_m), mae(y_true, y_pred_m),
                     r2(y_true, y_pred_a), mae(y_true, y_pred_a)))
    perf = pd.DataFrame(rows, columns=["L(년)","R2_월별","MAE_월별","R2_연평균","MAE_연평균"]).dropna().sort_values("L(년)")

    # ----- 곡선(두 방식 오버레이) + 간격/확대/스크롤 줌 -----
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf["L(년)"], y=perf["R2_월별"],
        mode="lines+markers+text",
        text=[f"{v:.4f}" for v in perf["R2_월별"]],
        textposition="top center",
        textfont=dict(size=11),
        name="R²(월별)"
    ))
    fig.add_trace(go.Scatter(
        x=perf["L(년)"], y=perf["R2_연평균"],
        mode="lines+markers",
        name="R²(연평균)", line=dict(dash="dot")
    ))
    # 최적 L(월별 기준)
    best_idx = perf["R2_월별"].idxmax()
    best_L, best_R2 = int(perf.loc[best_idx, "L(년)"]), float(perf.loc[best_idx, "R2_월별"])
    fig.add_vrect(x0=best_L-0.5, x1=best_L+0.5, fillcolor="#4CAF50", opacity=0.12, line_width=0,
                  annotation_text=f"최적 L(월별)={best_L}", annotation_position="top left")

    # R² 축을 상단 확대(기본 0.88~1.00), 데이터에 따라 자동 보정
    ymin_data = float(min(perf["R2_월별"].min(), perf["R2_연평균"].min()))
    ymax_data = float(max(perf["R2_월별"].max(), perf["R2_연평균"].max()))
    ymin = max(0.88, ymin_data - 0.02)
    ymax = min(1.0,  ymax_data + 0.01)

    fig.update_yaxes(title="R² (1에 가까울수록 유사)", range=[ymin, ymax], tick0=0.9, dtick=0.02)
    fig.update_xaxes(title=f"{target_year}년 예측 — ‘직전 L년’ 연속 평균 (월별 vs 연평균)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    tidy_layout(fig, title=f"R² 곡선 비교 — 월별(계절성 유지) vs 연평균(계절성 제거)", height=560)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["select", "lasso2d"]}
    )

    if show_table:
        table = perf.copy()
        table["ΔR2(월별-연평균)"] = table["R2_월별"] - table["R2_연평균"]
        table["비교구간"] = table["L(년)"].apply(lambda L: f"{target_year-L}~{target_year-1}")
        table = table[["L(년)","비교구간","R2_월별","R2_연평균","ΔR2(월별-연평균)","MAE_월별","MAE_연평균"]]
        st.dataframe(
            table.style.format({"R2_월별":"{:.4f}","R2_연평균":"{:.4f}",
                                "ΔR2(월별-연평균)":"{:.4f}","MAE_월별":"{:.3f}","MAE_연평균":"{:.3f}"}),
            use_container_width=True
        )

# ---------- 탭2 (1–4년 vs 5–8년만 표시) ----------
with tab2:
    colA, colB = st.columns([1,1])
    with colA:
        y_from = st.number_input("목표연도 시작", min_value=min_year+1, max_value=max_year-1,
                                 value=max(min_year+5, max_year-10))
    with colB:
        y_to   = st.number_input("목표연도 종료", min_value=y_from, max_value=max_year, value=max_year)

    # (Y,L) R² 매트릭스 — 후보는 모두 "연속 구간"
    mat_rows=[]
    for Y in range(int(y_from), int(y_to)+1):
        months_Y, y_true_Y = build_y_true(df, Y)
        for L in range(1, 11):
            start = Y - L
            if start < min_year:
                mat_rows.append((Y,L,np.nan)); continue
            y_pred = build_pred_monthly(df, start, Y-1, months_Y)
            mat_rows.append((Y, L, r2(y_true_Y, y_pred)))
    mat = pd.DataFrame(mat_rows, columns=["Y","L","R2"])

    # 연도별 최적 L
    best_per_Y = mat.loc[mat.groupby("Y")["R2"].idxmax()][["Y","L","R2"]].dropna().sort_values("Y")

    # (선택) 도넛: 최근/중간/장기 기본 분포는 유지
    best_per_Y["구분"] = np.where(best_per_Y["L"]<=3, "최근(1–3년)",
                           np.where(best_per_Y["L"]==4, "중간(4년)", "장기(5년+)"))
    dist = (best_per_Y["구분"].value_counts()
            .reindex(["최근(1–3년)","중간(4년)","장기(5년+)"])
            .fillna(0).reset_index())
    dist.columns = ["구분","연도수"]
    pie = px.pie(dist, names="구분", values="연도수", hole=0.35,
                 color="구분",
                 color_discrete_map={"최근(1–3년)":"#1976D2","중간(4년)":"#E53935","장기(5년+)":"#64B5F6"})
    pie.update_traces(textposition="inside", texttemplate="%{percent:.1%}\n(%{value}개 연도)")
    tidy_layout(pie, title="연도별 최적 L 분포(월별 방식 기준)", height=360)
    st.plotly_chart(pie, use_container_width=True)

    # ====== 1–4년 vs 5–8년: 범위 집계/KPI/막대 ======
    cnt_1_4 = int((best_per_Y["L"].between(1, 4)).sum())
    cnt_5_8 = int((best_per_Y["L"].between(5, 8)).sum())
    total2  = max(1, cnt_1_4 + cnt_5_8)
    def pct2(v): return v / total2 * 100.0

    c4, c5 = st.columns(2)
    c4.metric("최적 L: 1–4년", f"{cnt_1_4}개 연도", f"{pct2(cnt_1_4):.1f}%")
    c5.metric("최적 L: 5–8년", f"{cnt_5_8}개 연도", f"{pct2(cnt_5_8):.1f}%")

    range_df2 = pd.DataFrame({"구간": ["1–4년", "5–8년"], "연도수": [cnt_1_4, cnt_5_8]})
    bar2 = px.bar(range_df2, x="구간", y="연도수", text="연도수")
    bar2.update_traces(textposition="outside")
    bar2.update_layout(yaxis_title="연도수", xaxis_title="최적 L 범위")
    tidy_layout(bar2, title="최적 L 범위별(1–4년 vs 5–8년) 연도수", height=340)
    st.plotly_chart(bar2, use_container_width=True)

    # 최적 L 추이 + 히트맵
    fig_bestL = go.Figure()
    fig_bestL.add_hrect(y0=0.5, y1=3.5, fillcolor="#E3F2FD", opacity=0.35, line_width=0)
    fig_bestL.add_hrect(y0=3.5, y1=4.5, fillcolor="#FFEBEE", opacity=0.35, line_width=0)
    fig_bestL.add_hrect(y0=4.5, y1=10.5, fillcolor="#E8F5E9", opacity=0.25, line_width=0)
    fig_bestL.add_trace(go.Scatter(x=best_per_Y["Y"], y=best_per_Y["L"],
                                   mode="lines+markers+text",
                                   text=[str(int(v)) for v in best_per_Y["L"]],
                                   textposition="top center",
                                   name="최적 L(월별)"))
    fig_bestL.add_hline(y=3, line_dash="dot", line_color="#888")
    fig_bestL.update_yaxes(title="최적 L(년)", dtick=1, range=[1,10.1])
    fig_bestL.update_xaxes(title="목표연도 Y")
    tidy_layout(fig_bestL, title="연도별 최적 L 추이(낮을수록 최근 중심)")
    st.plotly_chart(fig_bestL, use_container_width=True)

    heat_df = mat.pivot(index="Y", columns="L", values="R2").sort_index()
    fig_hm = px.imshow(heat_df, labels=dict(x="L(년)", y="Y(목표연도)", color="R²"),
                       aspect="auto", color_continuous_scale="Blues", origin="lower")
    tidy_layout(fig_hm, title="세부 R² Heatmap — Y×L (연속 구간)", height=520)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.caption(f"(시트: {used_sheet}, 모드: {mode}, 규칙: ‘직전 L년 연속’만 후보)")
