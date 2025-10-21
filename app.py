# -*- coding: utf-8 -*-
# 최근 L년(1~10) 평균으로 목표연도 월별 기온을 예측했을 때의 적합도(R²) 비교 앱
# 탭 1) 단일연도: R² 곡선 + 최적 L 음영 + L=3 별표 + 설명문구
# 탭 2) 백테스트 요약: 목표연도 범위 × L=1..10 Heatmap(옵션)
#     + 승률/평균R²/ΔR²(부트스트랩 95% CI) + 최적L 분포(파이) + 최적L 추이(라인)
#
# 입력 엑셀: Wide([연도|1..12]) 또는 Long([날짜|평균기온]) 자동 인식

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="최근 L년 평균 vs 실제 — 단일연도 & 백테스트", layout="wide")

# ----------------- 파서 (Wide/Long 자동) -----------------
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
            if str(c).strip().lower() in ["날짜","date","일자","일시","dt"]:
                date_col = c; break
        if date_col is None:
            for c in body.columns:
                s = pd.to_datetime(body[c], errors="coerce")
                if s.notna().sum() >= max(12, int(len(s)*0.3)):
                    date_col = c; break
        if date_col is None: continue
        val_col = None
        for c in body.columns:
            if str(c).strip() in ["평균기온","기온","temp","temperature"]:
                val_col = c; break
        if val_col is None:
            nums = [c for c in body.columns if c!=date_col and pd.to_numeric(body[c], errors="coerce").notna().sum()>=max(12, int(len(body)*0.3))]
            if nums: val_col = nums[0]
        if val_col is None: continue

        df = body[[date_col, val_col]].copy().rename(columns={date_col:"date", val_col:"temp"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df = df.dropna(subset=["date","temp"])
        df["year"] = df["date"].dt.year
        df["month"]= df["date"].dt.month
        long = (df.groupby(["year","month"], as_index=False)["temp"].mean()
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

# ----------------- 공통 유틸 -----------------
def r2(y, yp):
    m = ~(np.isnan(y) | np.isnan(yp))
    y, yp = y[m], yp[m]
    if y.size < 2: return np.nan
    sse = np.sum((y-yp)**2)
    sst = np.sum((y - y.mean())**2)
    return float(1 - sse/sst) if sst>0 else np.nan

def month_mask(series_month, heating_mode: str):
    """heating_mode: 'all' / '11-3' / '10-3'"""
    if heating_mode == "all":
        return np.ones_like(series_month, dtype=bool)
    if heating_mode == "11-3":
        return series_month.isin([11,12,1,2,3])
    if heating_mode == "10-3":
        return series_month.isin([10,11,12,1,2,3])
    return np.ones_like(series_month, dtype=bool)

def build_y_true(df, Y:int, heating_mode:str):
    y_df = df.query("year == @Y").sort_values("month").copy()
    y_df = y_df[ month_mask(y_df["month"], heating_mode) ]
    months_order = y_df["month"].tolist()
    y_true = y_df["temp"].to_numpy()
    return months_order, y_true

def build_pred_from_train(df, start:int, end:int, months_order:list, heating_mode:str):
    train = df.query("year >= @start and year <= @end").copy()
    train = train[ month_mask(train["month"], heating_mode) ]
    preds=[]
    for m in months_order:
        x = train.loc[train["month"]==m, "temp"].to_numpy()
        preds.append(np.mean(x) if x.size>0 else np.nan)
    return np.array(preds, dtype=float)

def bootstrap_ci(values, B=4000, alpha=0.05, seed=1234):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(vals, size=n, replace=True)
        boots[i] = np.mean(sample)
    boots.sort()
    lo = boots[int((alpha/2)*B)]
    hi = boots[int((1-alpha/2)*B)-1]
    return float(np.mean(vals)), float(lo), float(hi)

# ----------------- 데이터 로딩 -----------------
default_path = Path("기온예측.xlsx")
uploaded = st.file_uploader("월별 평균기온 파일 업로드 (.xlsx)", type=["xlsx"])

df, used_sheet, mode = None, None, None
if uploaded:
    df, used_sheet, mode = load_excel_any(uploaded)
elif default_path.exists():
    df, used_sheet, mode = load_excel_any(default_path)

if df is None:
    st.error("엑셀에서 월별 평균기온을 찾지 못했어. 형식: A) [연도|1..12] 또는 B) [날짜|평균기온].")
    st.stop()

years = sorted(df["year"].unique())
min_year, max_year = int(min(years)), int(max(years))

tab1, tab2 = st.tabs(["단일연도", "백테스트 요약(최근1~3 vs 5+)"])

# ----------------- 탭1: 단일연도 -----------------
with tab1:
    c1, c2 = st.columns([1,1])
    with c1:
        target_year = st.number_input("대상연도(실제값 존재)", min_value=min_year+1, max_value=max_year, value=max_year, step=1)
    with c2:
        heating_mode = st.selectbox("평가 월 구간", options=[("all","전월(1–12월)"),("11-3","난방월(11–3월)"),("10-3","난방확장(10–3월)")],
                                    index=1, format_func=lambda x: x[1])[0]

    ex_start, ex_end = target_year-3, target_year-1
    st.markdown(f"**설명:** 예를 들어 **{target_year}년** 예측에 **최근 3년 평균**을 쓰면 = **{ex_start}~{ex_end}** 월평균으로 {target_year}년을 추정한다는 뜻.")
    st.markdown("### R² 곡선 — 직전 L년 평균(1~10)")

    months_order, y_true = build_y_true(df, target_year, heating_mode)
    end_year = target_year - 1

    rows=[]
    for L in range(1, 11):
        start = target_year - L
        if start < min_year: continue
        y_pred = build_pred_from_train(df, start, end_year, months_order, heating_mode)
        rows.append((L, r2(y_true, y_pred)))
    perf = pd.DataFrame(rows, columns=["L(년)","R2"]).dropna().sort_values("L(년)")

    if perf.empty:
        st.warning("계산 가능한 L이 없음.")
    else:
        best_row = perf.loc[perf["R2"].idxmax()]
        best_L, best_R2 = int(best_row["L(년)"]), float(best_row["R2"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf["L(년)"], y=perf["R2"], mode="lines+markers+text",
                                 text=[f"{v:.4f}" for v in perf["R2"]],
                                 textposition="top center", name="R²"))
        fig.add_vrect(x0=best_L-0.5, x1=best_L+0.5, fillcolor="#4CAF50", opacity=0.15, line_width=0,
                      annotation_text=f"최적 L={best_L}", annotation_position="top left")
        if 3 in perf["L(년)"].values:
            r2_3 = perf.loc[perf["L(년)"]==3, "R2"].values[0]
            fig.add_trace(go.Scatter(x=[3], y=[r2_3], mode="markers",
                                     marker=dict(size=14, symbol="star"), name="L=3(최근3년)"))
        ymin = max(0.0, float(perf["R2"].min())-0.01); ymax = min(1.0, float(perf["R2"].max())+0.01)
        fig.update_yaxes(title="R² (1에 가까울수록 좋음)", range=[ymin, ymax])
        fig.update_xaxes(title=f"직전 L년 평균 (L=1~10), 대상연도={target_year} → 학습: [Y-L .. Y-1]")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        mode_text = {"all":"전월", "11-3":"난방월", "10-3":"난방확장"}[heating_mode]
        trailing = " → ‘최근 3년’이 최적." if best_L==3 else ""
        st.success(f"요약({mode_text}): **{target_year}년** 예측에서 **직전 {best_L}년 평균**이 최고(R²={best_R2:.4f}).{trailing}")

# ----------------- 탭2: 백테스트 요약 -----------------
with tab2:
    colA, colB, colC = st.columns([1,1,1.3])
    with colA:
        y_from = st.number_input("목표연도 시작", min_value=min_year+1, max_value=max_year-1, value=max(min_year+5, max_year-10))
    with colB:
        y_to   = st.number_input("목표연도 종료", min_value=y_from, max_value=max_year, value=max_year)
    with colC:
        show_heat = st.checkbox("Heatmap(Y×L) 보기", value=True)

    colD, _ = st.columns([1,2])
    with colD:
        heating_mode_bt = st.selectbox("평가 월 구간(백테스트)", options=[("all","전월(1–12월)"),("11-3","난방월(11–3월)"),("10-3","난방확장(10–3월)")],
                                       index=1, format_func=lambda x: x[1])[0]

    # (Y, L) R² 매트릭스
    mat_rows=[]
    for Y in range(int(y_from), int(y_to)+1):
        months_order_Y, y_true_Y = build_y_true(df, Y, heating_mode_bt)
        endY = Y-1
        for L in range(1, 11):
            start = Y - L
            if start < min_year:
                mat_rows.append((Y, L, np.nan))
                continue
            y_pred = build_pred_from_train(df, start, endY, months_order_Y, heating_mode_bt)
            mat_rows.append((Y, L, r2(y_true_Y, y_pred)))
    mat = pd.DataFrame(mat_rows, columns=["Y","L","R2"])

    # 최근(1~3) vs 장기(5+) 비교
    recent = (mat.query("L<=3").groupby("Y")["R2"].max()).rename("R2_recent")
    long   = (mat.query("L>=5").groupby("Y")["R2"].max()).rename("R2_long")
    comp = pd.concat([recent, long], axis=1).dropna().reset_index()
    comp["Δ"] = comp["R2_recent"] - comp["R2_long"]
    comp["win_recent"] = (comp["Δ"] >= 0).astype(int)

    win_rate = comp["win_recent"].mean() if len(comp) else np.nan
    avg_recent = comp["R2_recent"].mean() if len(comp) else np.nan
    avg_long   = comp["R2_long"].mean() if len(comp) else np.nan
    diff_avg   = (avg_recent - avg_long) if (pd.notna(avg_recent) and pd.notna(avg_long)) else np.nan

    # ΔR² 부트스트랩 CI
    mean_delta, lo, hi = (np.nan, np.nan, np.nan)
    if not comp.empty:
        mean_delta, lo, hi = bootstrap_ci(comp["Δ"].to_list(), B=4000, alpha=0.05)

    mode_text = {"all":"전월", "11-3":"난방월", "10-3":"난방확장"}[heating_mode_bt]
    st.markdown(
        f"**요약({mode_text} · {int(y_from)}–{int(y_to)}):** "
        f"최근 1–3년 평균이 장기(≥5년)보다 우수한 연도 비율 = **{(win_rate*100):.1f}%**  |  "
        f"평균 R²: 최근 **{avg_recent:.4f}**, 장기 **{avg_long:.4f}**  (Δ={diff_avg:+.4f})  |  "
        f"ΔR² 95% CI **[{lo:+.4f}, {hi:+.4f}]**"
    )

    # 평균 R² 비교 막대
    bar = pd.DataFrame({"그룹":["최근 1–3년","장기 5년+"],"평균 R²":[avg_recent, avg_long]})
    fig_bar = px.bar(bar, x="그룹", y="평균 R²", text="평균 R²")
    fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    ymin_bar = max(0.0, min(avg_recent if pd.notna(avg_recent) else 1.0,
                            avg_long if pd.notna(avg_long) else 1.0) - 0.02)
    fig_bar.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10), yaxis_range=[ymin_bar, 1.0])
    st.plotly_chart(fig_bar, use_container_width=True)

    # 연도별 비교 표
    if not comp.empty:
        comp["최근(최적 L)"] = comp["Y"].apply(
            lambda Y: mat.query("Y==@Y and L<=3").sort_values("R2", ascending=False).iloc[0]["L"]
        )
        comp["장기(최적 L)"] = comp["Y"].apply(
            lambda Y: mat.query("Y==@Y and L>=5").sort_values("R2", ascending=False).iloc[0]["L"]
        )
        disp = comp[["Y","최근(최적 L)","R2_recent","장기(최적 L)","R2_long","Δ"]].copy()
        disp.rename(columns={"Y":"목표연도","R2_recent":"최근 R²","R2_long":"장기 R²","Δ":"ΔR²(최근-장기)"}, inplace=True)
        disp["최근 R²"] = disp["최근 R²"].map(lambda x: f"{x:.4f}")
        disp["장기 R²"] = disp["장기 R²"].map(lambda x: f"{x:.4f}")
        disp["ΔR²(최근-장기)"] = disp["ΔR²(최근-장기)"].map(lambda x: f"{x:+.4f}")
        st.dataframe(disp, use_container_width=True)

    # 최적 L 분포(파이) + 최적 L 추이(라인)
    best_per_Y = mat.loc[mat.groupby("Y")["R2"].idxmax()][["Y","L"]].dropna().sort_values("Y")
    if not best_per_Y.empty:
        # 분포(파이)
        tmp = best_per_Y.copy()
        tmp["구분"] = np.where(tmp["L"]<=3, "최근(1–3년)",
                         np.where(tmp["L"]>=5, "장기(5년+)", "중간(4년)"))
        dist = tmp["구분"].value_counts().reindex(["최근(1–3년)","중간(4년)","장기(5년+)"]).fillna(0).reset_index()
        dist.columns = ["구분","연도수"]
        pie = px.pie(dist, names="구분", values="연도수", hole=0.35)
        pie.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(pie, use_container_width=True)

        # 추이(라인)
        fig_bestL = go.Figure()
        fig_bestL.add_trace(go.Scatter(
            x=best_per_Y["Y"], y=best_per_Y["L"],
            mode="lines+markers+text",
            text=[str(int(v)) for v in best_per_Y["L"]],
            textposition="top center",
            name="최적 L"
        ))
        fig_bestL.add_hline(y=3, line_dash="dot", line_color="#888")
        fig_bestL.update_yaxes(title="최적 L(년)", dtick=1, range=[1,10.1])
        fig_bestL.update_xaxes(title="목표연도 Y")
        fig_bestL.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                                title="연도별 최적 L 추이(낮을수록 최근 연도 중심)")
        st.plotly_chart(fig_bestL, use_container_width=True)

    # Heatmap (옵션)
    if show_heat:
        heat_df = mat.pivot(index="Y", columns="L", values="R2").sort_index()
        fig_hm = px.imshow(heat_df, labels=dict(x="L(년)", y="목표연도 Y", color="R²"),
                           aspect="auto", color_continuous_scale="Blues", origin="lower")
        fig_hm.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_hm, use_container_width=True)

    st.caption(f"(시트: {used_sheet}, 모드: {mode})")
