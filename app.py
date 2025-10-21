# -*- coding: utf-8 -*-
# 최근 L년 평균 vs 실제 — 균형 비교(후보수 동일) + 최적 L 추이 + 최근(1–3년) 최적 연도 시각화

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="최근 L년 평균 vs 실제 — 균형 비교", layout="wide")

# ========== 공통 레이아웃 ==========
def tidy_layout(fig, title=None, height=360):
    if title:
        fig.update_layout(title=title, title_pad=dict(t=28, l=6, r=6, b=6))
    fig.update_layout(height=height, margin=dict(l=70, r=30, t=80, b=60))
    return fig

# ========== 엑셀 파서(와이드/롱 자동) ==========
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

# ========== 유틸 ==========
def r2(y, yp):
    y = np.array(y, dtype=float); yp = np.array(yp, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(yp))
    y, yp = y[mask], yp[mask]
    if y.size < 2: return np.nan
    sse = np.sum((y-yp)**2); sst = np.sum((y-y.mean())**2)
    return float(1 - sse/sst) if sst>0 else np.nan

def month_mask(series_month, heating_mode: str):
    if heating_mode == "all":   # 전월
        return np.ones_like(series_month, dtype=bool)
    if heating_mode == "11-3":  # 난방월
        return series_month.isin([11,12,1,2,3])
    if heating_mode == "10-3":  # 난방확장
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
    vals = np.array(values, dtype=float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); n = len(vals)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(vals, size=n, replace=True)
        boots[i] = np.mean(sample)
    boots.sort()
    lo = boots[int((alpha/2)*B)]
    hi = boots[int((1-alpha/2)*B)-1]
    return float(np.mean(vals)), float(lo), float(hi)

# ========== 데이터 로딩 ==========
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

# ========== 탭 ==========
tab1, tab2 = st.tabs(["단일연도", "백테스트 요약(균형 비교)"])

# ========== 탭1: 단일연도 ==========
with tab1:
    c1, c2 = st.columns([1,1])
    with c1:
        target_year = st.number_input("대상연도(실제값 존재)", min_value=min_year+1, max_value=max_year, value=max_year)
    with c2:
        heating_mode = st.selectbox(
            "평가 월 구간",
            options=[("all","전월(1–12월)"),("11-3","난방월(11–3월)"),("10-3","난방확장(10–3월)")],
            index=0, format_func=lambda x: x[1]
        )[0]
    ex_start, ex_end = target_year-3, target_year-1
    st.caption(f"예: {target_year}년 예측에서 ‘최근 3년 평균’은 {ex_start}~{ex_end}의 월평균으로 {target_year}년을 추정한다는 뜻.")

    months_order, y_true = build_y_true(df, target_year, heating_mode)
    rows=[]
    for L in range(1, 11):
        start = target_year - L
        if start < min_year: continue
        y_pred = build_pred_from_train(df, start, target_year-1, months_order, heating_mode)
        rows.append((L, r2(y_true, y_pred)))
    perf = pd.DataFrame(rows, columns=["L(년)","R2"]).dropna().sort_values("L(년)")

    if not perf.empty:
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
        tidy_layout(fig, title="R² 곡선 — 직전 L년 평균(1~10)", height=520)
        st.plotly_chart(fig, use_container_width=True)

# ========== 탭2: 백테스트(균형 비교) ==========
with tab2:
    colA, colB, colC = st.columns([1,1,1.1])
    with colA:
        y_from = st.number_input("목표연도 시작", min_value=min_year+1, max_value=max_year-1, value=max(min_year+5, max_year-10))
    with colB:
        y_to   = st.number_input("목표연도 종료", min_value=y_from, max_value=max_year, value=max_year)
    with colC:
        show_heat = st.checkbox("Heatmap(Y×L) 보기(고급)", value=False)

    colD, colE = st.columns([1,1])
    with colD:
        heating_mode_bt = st.selectbox(
            "평가 월 구간(백테스트)",
            options=[("all","전월(1–12월)"),("11-3","난방월(11–3월)"),("10-3","난방확장(10–3월)")],
            index=0, format_func=lambda x: x[1]
        )[0]
    with colE:
        add_long_8_10 = st.checkbox("장기 확장(8–10년 포함)", value=False)

    # 균형 비교 안내(난방월 선택시)
    if heating_mode_bt == "11-3":
        st.warning("난방월(11–3월)은 장기 창이 유리해질 수 있어. 본 비교는 ‘후보 수 동일(3 vs 3)’로 편향을 최소화했어.", icon="⚠️")

    # (Y, L) R² 매트릭스
    mat_rows=[]
    for Y in range(int(y_from), int(y_to)+1):
        months_order_Y, y_true_Y = build_y_true(df, Y, heating_mode_bt)
        endY = Y-1
        for L in range(1, 11):
            start = Y - L
            if start < min_year:
                mat_rows.append((Y, L, np.nan)); continue
            y_pred = build_pred_from_train(df, start, endY, months_order_Y, heating_mode_bt)
            mat_rows.append((Y, L, r2(y_true_Y, y_pred)))
    mat = pd.DataFrame(mat_rows, columns=["Y","L","R2"])

    # ===== 균형 비교(후보 수 동일) =====
    recent_set = [1,2,3]
    long_set   = [5,6,7] if not add_long_8_10 else [5,6,7,8,9,10]
    recent_mean_by_Y = mat[mat["L"].isin(recent_set)].groupby("Y")["R2"].mean()
    long_mean_by_Y   = mat[mat["L"].isin(long_set)].groupby("Y")["R2"].mean()

    comp = pd.concat([recent_mean_by_Y.rename("R2_recent_mean"),
                      long_mean_by_Y.rename("R2_long_mean")], axis=1).dropna().reset_index()
    comp["Δ"] = comp["R2_recent_mean"] - comp["R2_long_mean"]
    win_rate = (comp["Δ"] >= 0).mean() if not comp.empty else np.nan

    avg_recent = comp["R2_recent_mean"].mean() if not comp.empty else np.nan
    avg_long   = comp["R2_long_mean"].mean()   if not comp.empty else np.nan
    diff_avg   = (avg_recent - avg_long) if (pd.notna(avg_recent) and pd.notna(avg_long)) else np.nan
    mean_delta, lo, hi = (np.nan, np.nan, np.nan)
    if not comp.empty:
        mean_delta, lo, hi = bootstrap_ci(comp["Δ"].to_list(), B=4000, alpha=0.05)

    mode_text = {"all":"전월","11-3":"난방월","10-3":"난방확장"}[heating_mode_bt]
    long_label = "5–7년" if not add_long_8_10 else "5–10년"
    st.markdown(
        f"**요약({mode_text} · {int(y_from)}–{int(y_to)} · 균형 비교: 최근1–3 vs 장기{long_label}):** "
        f"최근 1–3년 **평균**이 장기 **평균**보다 우수한 연도 비율 = **{(win_rate*100):.1f}%**  |  "
        f"평균 R²: 최근 **{avg_recent:.4f}**, 장기 **{avg_long:.4f}** (Δ={diff_avg:+.4f})  |  "
        f"ΔR² 95% CI **[{lo:+.4f}, {hi:+.4f}]**"
    )

    # 평균 R² 막대
    bar = pd.DataFrame({"그룹":[f"최근 1–3년(평균)", f"장기 {long_label}(평균)"], "평균 R²":[avg_recent, avg_long]})
    fig_bar = px.bar(bar, x="그룹", y="평균 R²", text="평균 R²", color="그룹",
                     color_discrete_map={f"최근 1–3년(평균)":"#1976D2", f"장기 {long_label}(평균)":"#64B5F6"})
    fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    ymin_bar = max(0.0, min(avg_recent if pd.notna(avg_recent) else 1.0,
                            avg_long if pd.notna(avg_long) else 1.0) - 0.02)
    tidy_layout(fig_bar, title="평균 R² 비교 — 균형 비교(후보 수 동일)")
    fig_bar.update_yaxes(range=[ymin_bar, 1.0])
    st.plotly_chart(fig_bar, use_container_width=True)

    # 연도별 표
    if not comp.empty:
        disp = comp[["Y","R2_recent_mean","R2_long_mean","Δ"]].copy()
        disp.rename(columns={"Y":"목표연도","R2_recent_mean":"최근 평균 R²","R2_long_mean":f"장기 평균 R²({long_label})","Δ":"ΔR²(최근-장기)"}, inplace=True)
        disp["최근 평균 R²"] = disp["최근 평균 R²"].map(lambda x: f"{x:.4f}")
        disp[f"장기 평균 R²({long_label})"] = disp[f"장기 평균 R²({long_label})"].map(lambda x: f"{x:.4f}")
        disp["ΔR²(최근-장기)"] = disp["ΔR²(최근-장기)"].map(lambda x: f"{x:+.4f}")
        st.dataframe(disp, use_container_width=True)

    # ===== 최적 L 분포(도넛) + KPI + '최근 최적 연도' 시각화 =====
    best_per_Y = mat.loc[mat.groupby("Y")["R2"].idxmax()][["Y","L"]].dropna().sort_values("Y")
    if not best_per_Y.empty:
        tmp = best_per_Y.copy()
        tmp["구분"] = np.where(tmp["L"]<=3, "최근(1–3년)",
                         np.where(tmp["L"]==4, "중간(4년)", "장기(5년+)"))
        dist = tmp["구분"].value_counts().reindex(["최근(1–3년)","중간(4년)","장기(5년+)"]).fillna(0).reset_index()
        dist.columns = ["구분","연도수"]

        # 도넛
        pie = px.pie(dist, names="구분", values="연도수", hole=0.35,
                     color="구분",
                     color_discrete_map={"최근(1–3년)":"#1976D2","중간(4년)":"#E53935","장기(5년+)":"#64B5F6"})
        pie.update_traces(textposition="inside", texttemplate="%{percent:.1%}\n(%{value}개 연도)")
        tidy_layout(pie, title="연도별 최적 L 분포(퍼센트 + 연도수)", height=360)
        st.plotly_chart(pie, use_container_width=True)

        # KPI
        total_years = dist["연도수"].sum()
        def pct(v): return 0 if total_years==0 else v/total_years*100.0
        recent_cnt = int(dist.loc[dist["구분"]=="최근(1–3년)", "연도수"].fillna(0).values[0])
        mid_cnt    = int(dist.loc[dist["구분"]=="중간(4년)",   "연도수"].fillna(0).values[0])
        long_cnt   = int(dist.loc[dist["구분"]=="장기(5년+)",  "연도수"].fillna(0).values[0])

        c1, c2, c3 = st.columns(3)
        c1.metric("최근(1–3년) 최적 연도비중", f"{recent_cnt}개 연도", f"{pct(recent_cnt):.1f}%")
        c2.metric("중간(4년) 최적 연도비중",   f"{mid_cnt}개 연도",    f"{pct(mid_cnt):.1f}%")
        c3.metric("장기(5년+) 최적 연도비중",  f"{long_cnt}개 연도",   f"{pct(long_cnt):.1f}%")

        # (A) 연도별 최적 L '추이' — 이전 버전 그래프 복원
        fig_bestL = go.Figure()
        fig_bestL.add_hrect(y0=0.5, y1=3.5, fillcolor="#E3F2FD", opacity=0.35, line_width=0)  # 최근 영역
        fig_bestL.add_hrect(y0=3.5, y1=4.5, fillcolor="#FFEBEE", opacity=0.35, line_width=0)  # 중간 영역
        fig_bestL.add_hrect(y0=4.5, y1=10.5, fillcolor="#E8F5E9", opacity=0.25, line_width=0) # 장기 영역
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
        tidy_layout(fig_bestL, title="연도별 최적 L 추이(낮을수록 최근 중심)")
        st.plotly_chart(fig_bestL, use_container_width=True)

        # (B) 도넛 하단: '최근(1–3년) 최적'에 속한 연도만 모아서 별도 시각화
        recent_years = tmp.loc[tmp["구분"]=="최근(1–3년)","Y"].astype(int).tolist()
        recent_df = pd.DataFrame({"Y": recent_years})
        recent_df["값"] = 1  # 표시용

        # 텍스트로도 바로 확인
        if recent_years:
            years_txt = ", ".join(map(str, recent_years))
            st.info(f"최근(1–3년) 최적 연도: **{years_txt}**", icon="ℹ️")

            # (B-1) 스트립/막대 스타일: 해당 연도만 점으로 표시
            fig_strip = go.Figure()
            fig_strip.add_trace(go.Scatter(
                x=recent_df["Y"], y=recent_df["값"],
                mode="markers+text",
                text=recent_df["Y"].astype(str),
                textposition="top center",
                marker=dict(size=12, color="#1976D2"),
                name="최근 최적 연도"
            ))
            fig_strip.update_yaxes(visible=False, range=[0.8, 1.2])
            fig_strip.update_xaxes(title="최근(1–3년) 최적으로 나온 연도", tickmode="linear")
            tidy_layout(fig_strip, title=f"최근(1–3년) 최적 ‘{len(recent_years)}개 연도’ 시각화", height=260)
            st.plotly_chart(fig_strip, use_container_width=True)

        else:
            st.warning("최근(1–3년) 최적으로 나온 연도가 없음.", icon="⚠️")

    # Heatmap(옵션)
    if show_heat:
        heat_df = mat.pivot(index="Y", columns="L", values="R2").sort_index()
        fig_hm = px.imshow(heat_df, labels=dict(x="L(년)", y="목표연도 Y", color="R²"),
                           aspect="auto", color_continuous_scale="Blues", origin="lower")
        tidy_layout(fig_hm, title="R² Heatmap — Y×L", height=520)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.caption(f"(시트: {used_sheet}, 모드: {mode})")
